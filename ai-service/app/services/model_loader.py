"""
Singleton model loader for ML models.
Loads MTCNN and FaceNet models once at startup.
"""

import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

logger = logging.getLogger(__name__)


class ModelLoader:
    """Manages loading and lifecycle of ML models."""

    def __init__(self):
        self.face_detector = None
        self.facenet = None
        self.is_loaded = False

    def load_models(self):
        """Load all ML models."""
        from mtcnn import MTCNN
        from keras_facenet import FaceNet

        logger.info("Loading MTCNN Face Detection...")
        self.face_detector = MTCNN()

        logger.info("Loading FaceNet model...")
        self.facenet = FaceNet()

        self.is_loaded = True
        logger.info("All models loaded successfully.")

    def cleanup(self):
        """Release resources."""
        self.face_detector = None
        self.facenet = None
        self.is_loaded = False


model_loader = ModelLoader()
