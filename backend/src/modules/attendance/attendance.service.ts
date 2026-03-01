import path from 'path';
import fs from 'fs';
import ExcelJS from 'exceljs';
import { prisma } from '../../config/database';
import { config } from '../../config';
import { NotFoundError, BadRequestError, ForbiddenError } from '../../shared/exceptions';
import { SessionExistsError, SessionLockedError } from '../../shared/exceptions';
import { ParsedPagination, notDeleted } from '../../shared/utils/pagination';
import { aiServiceClient, RecognitionResult } from '../../integrations/ai-service/ai-service.client';
import { logger } from '../../config/logger';

const DAY_MAP: Record<number, string> = {
  0: 'SUNDAY',
  1: 'MONDAY',
  2: 'TUESDAY',
  3: 'WEDNESDAY',
  4: 'THURSDAY',
  5: 'FRIDAY',
  6: 'SATURDAY',
};

export class AttendanceService {
  // ── Teacher Schedule ──────────────────────────────────────────────────

  async getTodaySchedule(teacherId: string) {
    const today = DAY_MAP[new Date().getDay()];

    const teacher = await prisma.teacher.findFirst({
      where: { OR: [{ id: teacherId }, { userId: teacherId }], ...notDeleted() },
    });
    if (!teacher) throw new NotFoundError('Teacher not found');

    const schedules = await prisma.schedule.findMany({
      where: {
        courseOffering: {
          teacherAssignments: { some: { teacherId: teacher.id } },
        },
        dayOfWeek: today,
        isActive: true,
        ...notDeleted(),
      },
      include: {
        courseOffering: {
          include: {
            course: true,
            session: true,
            _count: { select: { studentEnrollments: true } },
          },
        },
        classroom: true,
      },
      orderBy: { startTime: 'asc' },
    });

    // Check which ones already have attendance sessions today
    const todayStart = new Date();
    todayStart.setHours(0, 0, 0, 0);
    const todayEnd = new Date();
    todayEnd.setHours(23, 59, 59, 999);

    const existingSessions = await prisma.attendanceSession.findMany({
      where: {
        teacherId: teacher.id,
        sessionDate: { gte: todayStart, lte: todayEnd },
      },
      select: { scheduleId: true, status: true, id: true },
    });

    const sessionMap = new Map(
      existingSessions.map((s) => [s.scheduleId, { id: s.id, status: s.status }])
    );

    return schedules.map((s) => ({
      ...s,
      attendanceSession: sessionMap.get(s.id) || null,
    }));
  }

  async getWeeklySchedule(teacherId: string) {
    const teacher = await prisma.teacher.findFirst({
      where: { OR: [{ id: teacherId }, { userId: teacherId }], ...notDeleted() },
    });
    if (!teacher) throw new NotFoundError('Teacher not found');

    const schedules = await prisma.schedule.findMany({
      where: {
        courseOffering: {
          teacherAssignments: { some: { teacherId: teacher.id } },
        },
        isActive: true,
        ...notDeleted(),
      },
      include: {
        courseOffering: {
          include: { course: true, session: true },
        },
        classroom: true,
      },
      orderBy: [{ startTime: 'asc' }],
    });

    // Group by day
    const weekly: Record<string, typeof schedules> = {};
    for (const day of Object.values(DAY_MAP)) {
      weekly[day] = schedules.filter((s) => s.dayOfWeek === day);
    }
    return weekly;
  }

  async getClassDetails(scheduleId: string, teacherId: string) {
    const teacher = await prisma.teacher.findFirst({
      where: { OR: [{ id: teacherId }, { userId: teacherId }], ...notDeleted() },
    });
    if (!teacher) throw new NotFoundError('Teacher not found');

    const schedule = await prisma.schedule.findFirst({
      where: { id: scheduleId, ...notDeleted() },
      include: {
        courseOffering: {
          include: {
            course: true,
            session: true,
            teacherAssignments: { where: { teacherId: teacher.id } },
            studentEnrollments: {
              where: { status: 'enrolled' },
              include: {
                student: {
                  select: {
                    id: true,
                    registrationNumber: true,
                    firstName: true,
                    lastName: true,
                    email: true,
                    profileImagePath: true,
                  },
                },
              },
            },
          },
        },
        classroom: true,
      },
    });

    if (!schedule) throw new NotFoundError('Schedule not found');
    if (schedule.courseOffering.teacherAssignments.length === 0) {
      throw new ForbiddenError('You are not assigned to this class');
    }

    return schedule;
  }

  // ── Attendance Sessions ───────────────────────────────────────────────

  async startSession(scheduleId: string, userId: string) {
    const teacher = await prisma.teacher.findFirst({
      where: { OR: [{ userId }], ...notDeleted() },
    });
    if (!teacher) throw new NotFoundError('Teacher not found');

    const schedule = await prisma.schedule.findFirst({
      where: { id: scheduleId, ...notDeleted() },
      include: {
        courseOffering: {
          include: {
            teacherAssignments: { where: { teacherId: teacher.id } },
          },
        },
      },
    });

    if (!schedule) throw new NotFoundError('Schedule not found');
    if (schedule.courseOffering.teacherAssignments.length === 0) {
      throw new ForbiddenError('You are not assigned to this class');
    }

    // Check if session already exists for today
    const todayStart = new Date();
    todayStart.setHours(0, 0, 0, 0);
    const todayEnd = new Date();
    todayEnd.setHours(23, 59, 59, 999);

    const existing = await prisma.attendanceSession.findFirst({
      where: {
        scheduleId,
        sessionDate: { gte: todayStart, lte: todayEnd },
      },
    });

    if (existing) {
      if (existing.status === 'SUBMITTED' || existing.status === 'FINALIZED') {
        throw new SessionLockedError();
      }
      throw new SessionExistsError(existing.id);
    }

    const session = await prisma.attendanceSession.create({
      data: {
        scheduleId,
        courseOfferingId: schedule.courseOfferingId,
        teacherId: teacher.id,
        classroomId: schedule.classroomId,
        sessionDate: new Date(),
        actualStartTime: new Date(),
        status: 'IN_PROGRESS',
        markedById: userId,
      },
      include: {
        courseOffering: { include: { course: true } },
        classroom: true,
      },
    });

    // Pre-populate attendance records as ABSENT for all enrolled students
    const enrollments = await prisma.studentEnrollment.findMany({
      where: {
        courseOfferingId: schedule.courseOfferingId,
        status: 'enrolled',
      },
    });

    if (enrollments.length > 0) {
      await prisma.attendanceRecord.createMany({
        data: enrollments.map((e) => ({
          attendanceSessionId: session.id,
          studentId: e.studentId,
          enrollmentId: e.id,
          status: 'ABSENT',
          markedBy: 'automatic',
        })),
      });
    }

    return session;
  }

  async processImage(sessionId: string, imagePath: string, userId: string, threshold?: number) {
    const session = await prisma.attendanceSession.findFirst({
      where: { id: sessionId },
      include: {
        courseOffering: {
          include: {
            studentEnrollments: {
              where: { status: 'enrolled' },
              include: { student: true },
            },
          },
        },
      },
    });

    if (!session) throw new NotFoundError('Attendance session not found');
    if (session.status === 'SUBMITTED' || session.status === 'FINALIZED') {
      throw new SessionLockedError();
    }

    // Verify teacher owns this session
    const teacher = await prisma.teacher.findFirst({
      where: { userId, ...notDeleted() },
    });
    if (!teacher || session.teacherId !== teacher.id) {
      throw new ForbiddenError('You do not own this attendance session');
    }

    const enrolledStudentIds = session.courseOffering.studentEnrollments.map(
      (e) => e.studentId
    );

    let recognitionResult: RecognitionResult;

    try {
      recognitionResult = await aiServiceClient.recognizeFaces(
        imagePath,
        enrolledStudentIds,
        threshold || 0.6
      );
    } catch (error: any) {
      // If AI service is down, store the image but don't process
      logger.warn('AI service unavailable, storing image for manual processing');

      // Save attendance image record
      await prisma.attendanceImage.create({
        data: {
          attendanceSessionId: sessionId,
          originalImagePath: imagePath,
          facesDetected: 0,
          facesRecognized: 0,
        },
      });

      return {
        aiServiceAvailable: false,
        message: 'AI service unavailable. Image saved. Please mark attendance manually.',
        imageSaved: true,
      };
    }

    // Move image to permanent storage
    const year = new Date().getFullYear().toString();
    const month = (new Date().getMonth() + 1).toString().padStart(2, '0');
    const permDir = path.join(config.storage.path, 'attendance', year, month, sessionId);

    if (!fs.existsSync(permDir)) {
      fs.mkdirSync(permDir, { recursive: true });
    }

    const permPath = path.join(permDir, `original${path.extname(imagePath)}`);
    fs.copyFileSync(imagePath, permPath);
    fs.unlinkSync(imagePath); // Remove temp file

    // Save attendance image record
    const attendanceImage = await prisma.attendanceImage.create({
      data: {
        attendanceSessionId: sessionId,
        originalImagePath: permPath,
        annotatedImagePath: recognitionResult.annotatedImagePath || null,
        facesDetected: recognitionResult.facesDetected,
        facesRecognized: recognitionResult.facesRecognized,
      },
    });

    // Update attendance records based on recognition
    for (const recognized of recognitionResult.recognizedStudents) {
      await prisma.attendanceRecord.updateMany({
        where: {
          attendanceSessionId: sessionId,
          studentId: recognized.studentId,
        },
        data: {
          status: 'PRESENT',
          markedBy: 'automatic',
          recognitionConfidence: recognized.confidence,
          recognitionDistance: recognized.distance,
          attendanceImageId: attendanceImage.id,
          faceLocation: JSON.stringify(recognized.faceLocation),
          checkInTime: new Date(),
        },
      });
    }

    // Update session detection method
    await prisma.attendanceSession.update({
      where: { id: sessionId },
      data: {
        detectionMethod: 'AI_RECOGNITION',
        recognitionThreshold: threshold || 0.6,
      },
    });

    // Return results with current attendance state
    const records = await prisma.attendanceRecord.findMany({
      where: { attendanceSessionId: sessionId },
      include: {
        student: {
          select: { id: true, registrationNumber: true, firstName: true, lastName: true },
        },
      },
    });

    return {
      aiServiceAvailable: true,
      facesDetected: recognitionResult.facesDetected,
      facesRecognized: recognitionResult.facesRecognized,
      processingTimeMs: recognitionResult.processingTimeMs,
      annotatedImageBase64: recognitionResult.annotatedImageBase64 || null,
      metrics: recognitionResult.metrics || null,
      recognizedStudents: recognitionResult.recognizedStudents || [],
      unknownFaces: recognitionResult.unknownFaces || [],
      attendanceRecords: records,
    };
  }

  async updateStudentAttendance(
    sessionId: string,
    studentId: string,
    status: string,
    userId: string,
    notes?: string
  ) {
    const session = await this.verifySessionOwnership(sessionId, userId);

    if (session.status === 'SUBMITTED' || session.status === 'FINALIZED') {
      throw new SessionLockedError();
    }

    const validStatuses = ['PRESENT', 'ABSENT', 'LATE', 'EXCUSED'];
    if (!validStatuses.includes(status)) {
      throw new BadRequestError(`Invalid status. Must be one of: ${validStatuses.join(', ')}`);
    }

    const record = await prisma.attendanceRecord.findFirst({
      where: { attendanceSessionId: sessionId, studentId },
    });

    if (!record) throw new NotFoundError('Attendance record not found');

    return prisma.attendanceRecord.update({
      where: { id: record.id },
      data: {
        status,
        markedBy: 'manual',
        notes,
      },
      include: {
        student: {
          select: { id: true, registrationNumber: true, firstName: true, lastName: true },
        },
      },
    });
  }

  async bulkUpdateAttendance(
    sessionId: string,
    updates: Array<{ studentId: string; status: string; notes?: string }>,
    userId: string
  ) {
    const session = await this.verifySessionOwnership(sessionId, userId);

    if (session.status === 'SUBMITTED' || session.status === 'FINALIZED') {
      throw new SessionLockedError();
    }

    const results = await Promise.all(
      updates.map(async (update) => {
        const record = await prisma.attendanceRecord.findFirst({
          where: { attendanceSessionId: sessionId, studentId: update.studentId },
        });
        if (!record) return null;

        return prisma.attendanceRecord.update({
          where: { id: record.id },
          data: {
            status: update.status,
            markedBy: 'manual',
            notes: update.notes,
          },
        });
      })
    );

    return results.filter(Boolean);
  }

  async submitSession(sessionId: string, userId: string) {
    const session = await this.verifySessionOwnership(sessionId, userId);

    if (session.status === 'SUBMITTED' || session.status === 'FINALIZED') {
      throw new SessionLockedError();
    }

    return prisma.attendanceSession.update({
      where: { id: sessionId },
      data: {
        status: 'SUBMITTED',
        submittedAt: new Date(),
        actualEndTime: new Date(),
      },
      include: {
        courseOffering: { include: { course: true } },
        attendanceRecords: {
          include: {
            student: {
              select: { id: true, registrationNumber: true, firstName: true, lastName: true },
            },
          },
        },
        _count: {
          select: { attendanceRecords: true },
        },
      },
    });
  }

  async getSessionDetails(sessionId: string, userId: string) {
    const teacher = await prisma.teacher.findFirst({
      where: { userId, ...notDeleted() },
    });

    const session = await prisma.attendanceSession.findFirst({
      where: { id: sessionId },
      include: {
        courseOffering: { include: { course: true, session: true } },
        classroom: true,
        schedule: true,
        attendanceRecords: {
          include: {
            student: {
              select: {
                id: true,
                registrationNumber: true,
                firstName: true,
                lastName: true,
                email: true,
              },
            },
          },
          orderBy: { student: { lastName: 'asc' } },
        },
        attendanceImages: true,
      },
    });

    if (!session) throw new NotFoundError('Attendance session not found');

    // Teachers can only view their own sessions
    if (teacher && session.teacherId !== teacher.id) {
      throw new ForbiddenError('You do not have access to this session');
    }

    // Compute summary stats
    const summary = {
      total: session.attendanceRecords.length,
      present: session.attendanceRecords.filter((r) => r.status === 'PRESENT').length,
      absent: session.attendanceRecords.filter((r) => r.status === 'ABSENT').length,
      late: session.attendanceRecords.filter((r) => r.status === 'LATE').length,
      excused: session.attendanceRecords.filter((r) => r.status === 'EXCUSED').length,
    };

    return { ...session, summary };
  }

  async getAttendanceHistory(
    userId: string,
    pagination: ParsedPagination,
    filters: {
      courseOfferingId?: string;
      status?: string;
      dateFrom?: string;
      dateTo?: string;
    }
  ) {
    const teacher = await prisma.teacher.findFirst({
      where: { userId, ...notDeleted() },
    });
    if (!teacher) throw new NotFoundError('Teacher not found');

    const where: Record<string, unknown> = { teacherId: teacher.id };

    if (filters.courseOfferingId) where.courseOfferingId = filters.courseOfferingId;
    if (filters.status) where.status = filters.status;
    if (filters.dateFrom || filters.dateTo) {
      where.sessionDate = {};
      if (filters.dateFrom) (where.sessionDate as any).gte = new Date(filters.dateFrom);
      if (filters.dateTo) (where.sessionDate as any).lte = new Date(filters.dateTo);
    }

    const [items, totalItems] = await Promise.all([
      prisma.attendanceSession.findMany({
        where: where as any,
        include: {
          courseOffering: { include: { course: true } },
          classroom: true,
          _count: { select: { attendanceRecords: true } },
        },
        skip: pagination.skip,
        take: pagination.limit,
        orderBy: { sessionDate: 'desc' },
      }),
      prisma.attendanceSession.count({ where: where as any }),
    ]);

    return { items, totalItems };
  }

  async downloadAttendanceSheet(sessionId: string, userId: string): Promise<{ buffer: Buffer; filename: string }> {
    const sessionData = await this.getSessionDetails(sessionId, userId);

    const workbook = new ExcelJS.Workbook();
    const sheet = workbook.addWorksheet('Attendance');

    const courseName = sessionData.courseOffering?.course?.name || 'Unknown Course';
    const date = new Date(sessionData.sessionDate).toLocaleDateString('en-GB');

    // Title row
    sheet.mergeCells('A1', 'D1');
    const titleCell = sheet.getCell('A1');
    titleCell.value = `${courseName} — Attendance Sheet (${date})`;
    titleCell.font = { bold: true, size: 14 };
    titleCell.alignment = { horizontal: 'center' };

    // Header row
    const headerRow = sheet.addRow(['#', 'Registration No.', 'Student Name', 'Status']);
    headerRow.eachCell((cell) => {
      cell.font = { bold: true, color: { argb: 'FFFFFFFF' } };
      cell.fill = { type: 'pattern', pattern: 'solid', fgColor: { argb: 'FF2563EB' } };
      cell.alignment = { horizontal: 'center' };
      cell.border = {
        top: { style: 'thin' },
        bottom: { style: 'thin' },
        left: { style: 'thin' },
        right: { style: 'thin' },
      };
    });

    // Data rows sorted by name
    const sorted = [...sessionData.attendanceRecords].sort((a: any, b: any) =>
      `${a.student.firstName} ${a.student.lastName}`.localeCompare(`${b.student.firstName} ${b.student.lastName}`)
    );

    sorted.forEach((record: any, idx: number) => {
      const status = record.status === 'PRESENT' || record.status === 'LATE' ? 'Present' : 'Absent';
      const row = sheet.addRow([
        idx + 1,
        record.student.registrationNumber,
        `${record.student.firstName} ${record.student.lastName}`,
        status,
      ]);

      row.eachCell((cell) => {
        cell.border = {
          top: { style: 'thin' },
          bottom: { style: 'thin' },
          left: { style: 'thin' },
          right: { style: 'thin' },
        };
        cell.alignment = { horizontal: 'center' };
      });

      // Color code status
      const statusCell = row.getCell(4);
      if (status === 'Present') {
        statusCell.font = { bold: true, color: { argb: 'FF16A34A' } };
      } else {
        statusCell.font = { bold: true, color: { argb: 'FFDC2626' } };
      }
    });

    // Column widths
    sheet.getColumn(1).width = 5;
    sheet.getColumn(2).width = 20;
    sheet.getColumn(3).width = 30;
    sheet.getColumn(4).width = 12;

    // Summary row
    sheet.addRow([]);
    const presentCount = sorted.filter((r: any) => r.status === 'PRESENT' || r.status === 'LATE').length;
    const summaryRow = sheet.addRow(['', '', 'Total Present:', `${presentCount}/${sorted.length}`]);
    summaryRow.getCell(3).font = { bold: true };
    summaryRow.getCell(4).font = { bold: true };

    const buffer = Buffer.from(await workbook.xlsx.writeBuffer());
    const safeCourseName = courseName.replace(/[^a-zA-Z0-9]/g, '_');
    const safeDate = date.replace(/\//g, '-');
    const filename = `Attendance_${safeCourseName}_${safeDate}.xlsx`;

    return { buffer, filename };
  }

  // ── Helpers ───────────────────────────────────────────────────────────

  private async verifySessionOwnership(sessionId: string, userId: string) {
    const teacher = await prisma.teacher.findFirst({
      where: { userId, ...notDeleted() },
    });
    if (!teacher) throw new NotFoundError('Teacher not found');

    const session = await prisma.attendanceSession.findFirst({
      where: { id: sessionId },
    });
    if (!session) throw new NotFoundError('Attendance session not found');

    if (session.teacherId !== teacher.id) {
      throw new ForbiddenError('You do not own this attendance session');
    }

    return session;
  }
}

export const attendanceService = new AttendanceService();
