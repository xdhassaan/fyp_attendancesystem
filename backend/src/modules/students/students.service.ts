import { prisma } from '../../config/database';
import { NotFoundError, ConflictError } from '../../shared/exceptions';
import { ParsedPagination, notDeleted } from '../../shared/utils/pagination';

export class StudentsService {
  async create(data: {
    registrationNumber: string;
    firstName: string;
    lastName: string;
    email?: string;
    phone?: string;
    departmentId?: string;
    batchId?: string;
  }) {
    return prisma.student.create({
      data: {
        ...data,
        isActive: true,
      },
      include: {
        department: true,
        batch: true,
        _count: { select: { faceImages: true } },
      },
    });
  }

  async findAll(
    pagination: ParsedPagination,
    filters: {
      search?: string;
      departmentId?: string;
      batchId?: string;
      isActive?: boolean;
    }
  ) {
    const where: Record<string, unknown> = { ...notDeleted() };

    if (filters.search) {
      where.OR = [
        { firstName: { contains: filters.search } },
        { lastName: { contains: filters.search } },
        { registrationNumber: { contains: filters.search } },
        { email: { contains: filters.search } },
      ];
    }
    if (filters.departmentId) where.departmentId = filters.departmentId;
    if (filters.batchId) where.batchId = filters.batchId;
    if (filters.isActive !== undefined) where.isActive = filters.isActive;

    const [items, totalItems] = await Promise.all([
      prisma.student.findMany({
        where: where as any,
        include: {
          department: true,
          batch: true,
          _count: { select: { faceImages: { where: notDeleted() } } },
        },
        skip: pagination.skip,
        take: pagination.limit,
        orderBy: { [pagination.sortBy]: pagination.sortOrder },
      }),
      prisma.student.count({ where: where as any }),
    ]);

    return { items, totalItems };
  }

  async findById(id: string) {
    const student = await prisma.student.findFirst({
      where: { id, ...notDeleted() },
      include: {
        department: true,
        batch: true,
        faceImages: { where: notDeleted(), orderBy: { createdAt: 'desc' } },
        enrollments: {
          where: { status: 'enrolled' },
          include: {
            courseOffering: {
              include: { course: true },
            },
          },
        },
        _count: {
          select: {
            faceImages: { where: notDeleted() },
            attendanceRecords: true,
          },
        },
      },
    });

    if (!student) throw new NotFoundError('Student not found');
    return student;
  }

  async update(id: string, data: {
    firstName?: string;
    lastName?: string;
    email?: string;
    phone?: string;
    departmentId?: string;
    batchId?: string;
    isActive?: boolean;
  }) {
    await this.findById(id); // throws if not found

    return prisma.student.update({
      where: { id },
      data,
      include: {
        department: true,
        batch: true,
      },
    });
  }

  async delete(id: string) {
    await this.findById(id);

    return prisma.student.update({
      where: { id },
      data: { deletedAt: new Date() },
    });
  }

  async addFaceImages(studentId: string, files: Express.Multer.File[]) {
    await this.findById(studentId);

    const images = await Promise.all(
      files.map((file) =>
        prisma.studentFaceImage.create({
          data: {
            studentId,
            imagePath: file.path,
            imageFilename: file.originalname,
            fileSizeBytes: file.size,
          },
        })
      )
    );

    return images;
  }
}

export const studentsService = new StudentsService();
