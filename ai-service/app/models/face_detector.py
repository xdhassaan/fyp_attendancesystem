"""
Face detection using MTCNN.
Extracted and adapted from the original face_recognition_attendance.py.
Enhanced with contrast/denoise preprocessing and quality assessment.
"""

import numpy as np
import cv2
import logging
from typing import List, Dict, Optional, Tuple

from app.services.model_loader import model_loader

logger = logging.getLogger(__name__)

# FaceNet expected input size
FACENET_INPUT_SIZE = (160, 160)
FACE_MARGIN = 0.35


def detect_faces(image_rgb: np.ndarray) -> List[Dict]:
    """
    Detect faces using MTCNN.
    Returns list of dicts with box, confidence, left_eye, right_eye.
    """
    results = []

    detections = model_loader.face_detector.detect_faces(image_rgb)

    for detection in detections:
        box = detection["box"]  # [x, y, width, height]
        confidence = detection["confidence"]
        keypoints = detection.get("keypoints", {})

        # Clamp to image bounds
        h, w = image_rgb.shape[:2]
        x = max(0, box[0])
        y = max(0, box[1])
        box_w = min(box[2], w - x)
        box_h = min(box[3], h - y)

        left_eye = keypoints.get("left_eye")
        right_eye = keypoints.get("right_eye")

        results.append({
            "box": (x, y, box_w, box_h),
            "confidence": float(confidence),
            "left_eye": left_eye,
            "right_eye": right_eye,
        })

    return results


def align_face(image: np.ndarray, left_eye: Tuple, right_eye: Tuple) -> np.ndarray:
    """Align face based on eye positions."""
    try:
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        eye_center = (
            (left_eye[0] + right_eye[0]) // 2,
            (left_eye[1] + right_eye[1]) // 2,
        )

        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        h, w = image.shape[:2]
        aligned = cv2.warpAffine(
            image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        return aligned
    except Exception:
        return image


def add_margin(image: np.ndarray, box: Tuple, margin_percent: float = FACE_MARGIN):
    """Add margin around face with boundary checking."""
    h, w = image.shape[:2]
    x, y, face_w, face_h = box

    margin_w = int(face_w * margin_percent)
    margin_h = int(face_h * margin_percent)

    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(w, x + face_w + margin_w)
    y2 = min(h, y + face_h + margin_h)

    return image[y1:y2, x1:x2], (x1, y1, x2, y2)


def normalize_lighting(face_image: np.ndarray) -> np.ndarray:
    """Normalize lighting using CLAHE on LAB color space."""
    try:
        lab = cv2.cvtColor(face_image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    except Exception:
        return face_image


def enhance_contrast(face_image: np.ndarray) -> np.ndarray:
    """Enhance contrast using histogram stretching in YCrCb space."""
    try:
        ycrcb = cv2.cvtColor(face_image, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_min, y_max = np.percentile(y, [2, 98])
        if y_max - y_min > 0:
            y = np.clip(
                (y.astype(np.float32) - y_min) * 255.0 / (y_max - y_min), 0, 255
            ).astype(np.uint8)
        ycrcb = cv2.merge([y, cr, cb])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    except Exception:
        return face_image


def denoise_face(face_image: np.ndarray) -> np.ndarray:
    """Light denoising to reduce compression artifacts."""
    try:
        return cv2.fastNlMeansDenoisingColored(face_image, None, 3, 3, 7, 21)
    except Exception:
        return face_image


def assess_face_quality(
    face_image: np.ndarray, min_size: int = 40
) -> Tuple[bool, Dict]:
    """
    Assess face quality. Returns (is_acceptable, metrics).
    Checks resolution, blur (Laplacian variance), and brightness.
    """
    h, w = face_image.shape[:2]

    if h < min_size or w < min_size:
        return False, {"reason": "too_small", "size": (w, h)}

    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < 15.0:
        return False, {"reason": "blurry", "laplacian_var": float(laplacian_var)}

    mean_brightness = float(np.mean(gray))
    if mean_brightness < 30 or mean_brightness > 240:
        return False, {"reason": "bad_exposure", "brightness": mean_brightness}

    return True, {
        "laplacian_var": float(laplacian_var),
        "brightness": mean_brightness,
    }


def preprocess_face(
    image: np.ndarray,
    box: Tuple,
    left_eye: Optional[Tuple] = None,
    right_eye: Optional[Tuple] = None,
) -> Tuple[np.ndarray, Tuple]:
    """
    Full preprocessing pipeline:
    margin → alignment → CLAHE → contrast enhancement → denoise → resize.
    Returns (preprocessed_face, bounding_box_with_margin).
    """
    face_with_margin, new_box = add_margin(image, box, FACE_MARGIN)

    if left_eye is not None and right_eye is not None:
        x1, y1 = new_box[0], new_box[1]
        adjusted_left = (left_eye[0] - x1, left_eye[1] - y1)
        adjusted_right = (right_eye[0] - x1, right_eye[1] - y1)
        face_with_margin = align_face(face_with_margin, adjusted_left, adjusted_right)

    normalized = normalize_lighting(face_with_margin)
    enhanced = enhance_contrast(normalized)
    denoised = denoise_face(enhanced)
    resized = cv2.resize(denoised, FACENET_INPUT_SIZE, interpolation=cv2.INTER_CUBIC)

    return resized, new_box


def get_embedding(face_image: np.ndarray) -> np.ndarray:
    """Generate L2-normalized FaceNet embedding for a preprocessed face image."""
    raw = model_loader.facenet.embeddings([face_image])[0]
    norm = np.linalg.norm(raw)
    if norm > 0:
        raw = raw / norm
    return raw


def draw_annotations(
    image: np.ndarray,
    boxes: List[Tuple],
    names: List[str],
    confidences: List[float],
) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    img = image.copy()
    h, w = img.shape[:2]

    scale_factor = max(w, h) / 1920
    scale_factor = max(0.5, min(scale_factor, 3.0))

    box_thickness = max(2, int(4 * scale_factor))
    font_scale = max(0.6, 1.0 * scale_factor)
    font_thickness = max(1, int(2 * scale_factor))
    padding = max(5, int(10 * scale_factor))

    for idx, ((x1, y1, x2, y2), name) in enumerate(zip(boxes, names)):
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 200)

        cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thickness)

        if name != "Unknown" and idx < len(confidences):
            label = f"{name} ({confidences[idx]:.0f}%)"
        else:
            label = name

        (text_w, text_h), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
        )

        label_y1 = max(0, y1 - text_h - 2 * padding)
        cv2.rectangle(img, (x1, label_y1), (x1 + text_w + 2 * padding, y1), color, -1)
        cv2.putText(
            img,
            label,
            (x1 + padding, y1 - padding),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return img
