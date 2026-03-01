import { Router } from 'express';
import { testingController } from './testing.controller';
import { authMiddleware } from '../../middleware/auth.middleware';
import { testerOnly } from '../../middleware/rbac.middleware';
import { uploadTestingImages } from '../../middleware/upload.middleware';

const router = Router();

router.use(authMiddleware, testerOnly);

router.post('/recognize', uploadTestingImages, testingController.recognize);
router.get('/students', testingController.getStudents);
router.post('/download-excel', testingController.downloadExcel);

export default router;
