import axios, { AxiosInstance } from 'axios';
import FormData from 'form-data';
import fs from 'fs';
import { config } from '../../config';
import { logger } from '../../config/logger';
import { AppError } from '../../shared/exceptions';

export interface RecognizedFace {
  studentId: string;
  registrationNumber: string;
  name: string;
  confidence: number;
  distance: number;
  matchMethod?: string;
  faceLocation: { x1: number; y1: number; x2: number; y2: number };
}

export interface RecognitionResult {
  facesDetected: number;
  facesRecognized: number;
  recognizedStudents: RecognizedFace[];
  unknownFaces: Array<{
    faceLocation: { x1: number; y1: number; x2: number; y2: number };
    confidence: number;
  }>;
  annotatedImagePath?: string;
  annotatedImageBase64?: string;
  processingTimeMs: number;
  metrics?: {
    threshold: number;
    distanceMetric: string;
    svmAvailable: boolean;
    imageWidth: number;
    imageHeight: number;
    enrolledStudents: number;
    studentsWithEncodings: number;
    facesDetected: number;
    facesRecognized: number;
    unknownFaces: number;
    skippedQuality: number;
    recognitionRate: number;
    avgDistance: number | null;
    minDistance: number | null;
    maxDistance: number | null;
    avgConfidence: number | null;
    processingTimeMs: number;
  };
}

export interface EncodingResult {
  studentId: string;
  encodingsGenerated: number;
  success: boolean;
  error?: string;
}

export interface HealthStatus {
  status: string;
  modelLoaded: boolean;
  version: string;
}

class AIServiceClient {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: config.aiService.url,
      timeout: config.aiService.timeout,
      headers: {
        ...(config.aiService.apiKey && { 'X-API-Key': config.aiService.apiKey }),
      },
    });
  }

  async healthCheck(): Promise<HealthStatus> {
    try {
      const response = await this.client.get('/health');
      return response.data;
    } catch (error) {
      logger.error('AI service health check failed:', error);
      throw new AppError('AI service is unavailable', 503, 'AI_SERVICE_UNAVAILABLE');
    }
  }

  async recognizeFaces(
    imagePath: string,
    enrolledStudentIds: string[],
    threshold: number = 0.6
  ): Promise<RecognitionResult> {
    try {
      const formData = new FormData();
      formData.append('image', fs.createReadStream(imagePath));
      formData.append('student_ids', JSON.stringify(enrolledStudentIds));
      formData.append('threshold', threshold.toString());

      const response = await this.client.post('/api/v1/recognize', formData, {
        headers: formData.getHeaders(),
        timeout: config.aiService.timeout,
      });

      return response.data;
    } catch (error: any) {
      logger.error('Face recognition failed:', error?.message);
      if (error?.response?.status === 422) {
        throw new AppError(
          error.response.data?.detail || 'No faces detected in image',
          422,
          'FACE_DETECTION_FAILED'
        );
      }
      throw new AppError('Face recognition service failed', 502, 'AI_SERVICE_ERROR');
    }
  }

  async generateEncodings(
    studentId: string,
    imagePaths: string[]
  ): Promise<EncodingResult> {
    try {
      const formData = new FormData();
      formData.append('student_id', studentId);
      for (const imagePath of imagePaths) {
        formData.append('images', fs.createReadStream(imagePath));
      }

      const response = await this.client.post('/api/v1/encodings/generate', formData, {
        headers: formData.getHeaders(),
        timeout: config.aiService.timeout,
      });

      return response.data;
    } catch (error: any) {
      logger.error('Encoding generation failed:', error?.message);
      throw new AppError('Failed to generate face encodings', 502, 'AI_SERVICE_ERROR');
    }
  }

  async getAllStudentIds(): Promise<string[]> {
    try {
      const response = await this.client.get('/api/v1/encodings');
      return response.data.studentIds || [];
    } catch (error: any) {
      logger.error('Failed to get student IDs:', error?.message);
      return [];
    }
  }

  async getAllStudentsWithMetadata(): Promise<Array<{
    studentId: string;
    name: string;
    registrationNumber: string;
    encodingCount: number;
  }>> {
    try {
      const response = await this.client.get('/api/v1/encodings/students');
      return response.data.students || [];
    } catch (error: any) {
      logger.error('Failed to get students metadata:', error?.message);
      return [];
    }
  }

  async deleteEncodings(studentId: string): Promise<void> {
    try {
      await this.client.delete(`/api/v1/encodings/${studentId}`);
    } catch (error: any) {
      logger.warn(`Failed to delete encodings for student ${studentId}:`, error?.message);
    }
  }
}

export const aiServiceClient = new AIServiceClient();
