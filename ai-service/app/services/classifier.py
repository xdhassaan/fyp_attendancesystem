"""
SVM classifier for face recognition.
Trained on stored FaceNet embeddings; provides predict + probability.
Used as an ensemble with distance-based matching.
"""

import os
import logging
import pickle
from typing import Optional, Dict, List

import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CLASSIFIER_DIR = os.path.join(BASE_DIR, "models")
CLASSIFIER_PATH = os.path.join(CLASSIFIER_DIR, "svm_classifier.pkl")
LABEL_ENCODER_PATH = os.path.join(CLASSIFIER_DIR, "label_encoder.pkl")


class FaceClassifier:
    def __init__(self):
        self.svm: Optional[SVC] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.is_trained = False
        self._load_if_exists()

    def _load_if_exists(self):
        """Load pre-trained classifier if available."""
        if os.path.exists(CLASSIFIER_PATH) and os.path.exists(LABEL_ENCODER_PATH):
            try:
                with open(CLASSIFIER_PATH, "rb") as f:
                    self.svm = pickle.load(f)
                with open(LABEL_ENCODER_PATH, "rb") as f:
                    self.label_encoder = pickle.load(f)
                self.is_trained = True
                logger.info(
                    f"Loaded SVM classifier ({len(self.label_encoder.classes_)} classes)"
                )
            except Exception as e:
                logger.warning(f"Failed to load classifier: {e}")

    def train(self, encodings_by_student: Dict[str, np.ndarray]) -> Dict:
        """
        Train SVM on all student encodings.
        encodings_by_student: {student_id: np.ndarray of shape (N, 128)}
        """
        X, y = [], []
        for student_id, encs in encodings_by_student.items():
            if encs is not None and len(encs) > 0:
                # L2-normalize
                norms = np.linalg.norm(encs, axis=1, keepdims=True)
                norms = np.where(norms > 0, norms, 1)
                normed = encs / norms
                for enc in normed:
                    X.append(enc)
                    y.append(student_id)

        if len(set(y)) < 2:
            logger.warning("Need at least 2 students to train SVM")
            return {"success": False, "reason": "insufficient_classes"}

        X = np.array(X)
        y = np.array(y)

        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)

        self.svm = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
        )
        self.svm.fit(X, y_encoded)
        self.is_trained = True

        # Save
        os.makedirs(CLASSIFIER_DIR, exist_ok=True)
        with open(CLASSIFIER_PATH, "wb") as f:
            pickle.dump(self.svm, f)
        with open(LABEL_ENCODER_PATH, "wb") as f:
            pickle.dump(self.label_encoder, f)

        logger.info(f"SVM trained: {len(set(y))} classes, {len(X)} samples")
        return {
            "success": True,
            "classes": int(len(set(y))),
            "samples": int(len(X)),
        }

    def predict(
        self, embedding: np.ndarray, enrolled_ids: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """
        Predict student ID for a face embedding.
        Returns {studentId, probability} or None.
        """
        if not self.is_trained:
            return None

        # Normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        try:
            probs = self.svm.predict_proba([embedding])[0]
        except Exception:
            return None

        all_classes = self.label_encoder.classes_

        # If enrolled_ids provided, mask to only enrolled students
        if enrolled_ids:
            enrolled_set = set(enrolled_ids)
            mask = np.array([cls in enrolled_set for cls in all_classes])
            if not mask.any():
                return None
            # Zero out non-enrolled, renormalize
            masked_probs = probs * mask
            total = masked_probs.sum()
            if total > 0:
                masked_probs = masked_probs / total
            probs = masked_probs

        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        best_id = str(all_classes[best_idx])

        return {"studentId": best_id, "probability": best_prob}


face_classifier = FaceClassifier()
