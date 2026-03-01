"""
Encoding store - manages stored face encodings.
Uses a local JSON + numpy file approach for MVP.
Includes per-student stats for adaptive thresholds.
"""

import os
import json
import logging
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENCODINGS_DIR = os.path.join(BASE_DIR, "encodings")


class EncodingStore:
    """
    Manages face encodings storage.
    Each student has a directory with:
      - metadata.json (studentId, name, registrationNumber)
      - encodings.npy (numpy array of embeddings)
      - stats.json (intra-class distance stats for adaptive thresholds)
    """

    def __init__(self):
        os.makedirs(ENCODINGS_DIR, exist_ok=True)

    def save_encodings(
        self,
        student_id: str,
        encodings: List[np.ndarray],
        metadata: Optional[Dict] = None,
    ):
        """Save encodings for a student."""
        student_dir = os.path.join(ENCODINGS_DIR, student_id)
        os.makedirs(student_dir, exist_ok=True)

        # Save or append encodings
        enc_path = os.path.join(student_dir, "encodings.npy")
        if os.path.exists(enc_path):
            existing = np.load(enc_path)
            all_encodings = np.vstack([existing, np.array(encodings)])
        else:
            all_encodings = np.array(encodings)

        np.save(enc_path, all_encodings)

        # Save metadata
        if metadata:
            meta_path = os.path.join(student_dir, "metadata.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f)

        logger.info(
            f"Saved {len(encodings)} encodings for student {student_id} "
            f"(total: {len(all_encodings)})"
        )

    def get_encodings(self, student_id: str) -> Optional[Dict]:
        """Get encodings and metadata for a student."""
        student_dir = os.path.join(ENCODINGS_DIR, student_id)
        enc_path = os.path.join(student_dir, "encodings.npy")

        if not os.path.exists(enc_path):
            return None

        encodings = np.load(enc_path)
        metadata = {}

        meta_path = os.path.join(student_dir, "metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)

        return {
            "encodings": encodings.tolist() if isinstance(encodings, np.ndarray) else encodings,
            "encodings_np": encodings,
            **metadata,
        }

    def get_encodings_for_students(self, student_ids: List[str]) -> Dict[str, Dict]:
        """Get encodings for multiple students (only those who have them)."""
        result = {}
        for sid in student_ids:
            data = self.get_encodings(sid)
            if data:
                # Keep numpy arrays for computation
                result[sid] = {
                    "encodings": data["encodings_np"],
                    "registrationNumber": data.get("registrationNumber", ""),
                    "name": data.get("name", ""),
                }
        return result

    def delete_encodings(self, student_id: str) -> bool:
        """Delete all encodings for a student."""
        import shutil

        student_dir = os.path.join(ENCODINGS_DIR, student_id)
        if os.path.exists(student_dir):
            shutil.rmtree(student_dir)
            logger.info(f"Deleted encodings for student {student_id}")
            return True
        return False

    def get_all_student_ids(self) -> List[str]:
        """List all students that have stored encodings."""
        if not os.path.exists(ENCODINGS_DIR):
            return []
        return [
            d
            for d in os.listdir(ENCODINGS_DIR)
            if os.path.isdir(os.path.join(ENCODINGS_DIR, d))
        ]

    def compute_student_stats(self, student_id: str) -> Optional[Dict]:
        """Compute encoding statistics for adaptive thresholding."""
        data = self.get_encodings(student_id)
        if not data or data["encodings_np"] is None or len(data["encodings_np"]) < 2:
            return None

        encodings = data["encodings_np"]
        # L2-normalize
        norms = np.linalg.norm(encodings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        normed = encodings / norms

        # Pairwise cosine distances within this student
        similarities = np.dot(normed, normed.T)
        distances = 1 - similarities
        upper = distances[np.triu_indices_from(distances, k=1)]

        stats = {
            "mean_intra_dist": float(np.mean(upper)) if len(upper) > 0 else 0,
            "max_intra_dist": float(np.max(upper)) if len(upper) > 0 else 0,
            "std_intra_dist": float(np.std(upper)) if len(upper) > 0 else 0,
            "num_encodings": int(len(encodings)),
        }

        student_dir = os.path.join(ENCODINGS_DIR, student_id)
        stats_path = os.path.join(student_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f)

        return stats

    def get_student_threshold(self, student_id: str, base_threshold: float) -> float:
        """Get adaptive threshold for a student based on encoding spread."""
        student_dir = os.path.join(ENCODINGS_DIR, student_id)
        stats_path = os.path.join(student_dir, "stats.json")

        if not os.path.exists(stats_path):
            return base_threshold

        try:
            with open(stats_path, "r") as f:
                stats = json.load(f)
        except Exception:
            return base_threshold

        # Adaptive: base + margin based on intra-class spread
        intra_max = stats.get("max_intra_dist", 0)
        adaptive = base_threshold + 0.3 * intra_max
        # Clamp to reasonable range
        return min(adaptive, base_threshold * 1.4)


encoding_store = EncodingStore()
