"""
Face recognition service.
Matches detected faces against stored encodings using cosine distance + SVM ensemble.
"""

import os
import json
import logging
import time
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import numpy as np
import cv2

from app.models.face_detector import (
    detect_faces,
    preprocess_face,
    get_embedding,
    draw_annotations,
    assess_face_quality,
)
from app.services.encoding_store import encoding_store
from app.services.classifier import face_classifier

logger = logging.getLogger(__name__)


def recognize_faces_in_image(
    image_rgb: np.ndarray,
    enrolled_student_ids: List[str],
    threshold: float = 0.6,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Process a class image:
    1. Detect all faces
    2. Generate L2-normalized embeddings
    3. Match via cosine distance + SVM ensemble
    4. Return recognition results with metrics
    """
    start_time = time.time()

    # Detect faces
    faces = detect_faces(image_rgb)

    if not faces:
        return {
            "facesDetected": 0,
            "facesRecognized": 0,
            "recognizedStudents": [],
            "unknownFaces": [],
            "processingTimeMs": int((time.time() - start_time) * 1000),
        }

    # Load encodings for enrolled students only
    student_encodings = encoding_store.get_encodings_for_students(enrolled_student_ids)

    recognized_students = []
    unknown_faces = []
    boxes = []
    names = []
    confidences = []
    skipped_quality = 0
    rec_embeddings = []     # parallel to recognized_students — for re-matching during dedup
    rec_box_indices = []    # parallel to recognized_students — index into boxes/names/confidences

    for face in faces:
        box = face["box"]
        left_eye = face.get("left_eye")
        right_eye = face.get("right_eye")

        try:
            # Preprocess and get embedding
            preprocessed, new_box = preprocess_face(image_rgb, box, left_eye, right_eye)

            # Quality gate
            is_acceptable, quality_info = assess_face_quality(preprocessed)
            if not is_acceptable:
                logger.debug(f"Skipping low-quality face: {quality_info}")
                skipped_quality += 1
                unknown_faces.append({
                    "faceLocation": {
                        "x1": int(new_box[0]),
                        "y1": int(new_box[1]),
                        "x2": int(new_box[2]),
                        "y2": int(new_box[3]),
                    },
                    "confidence": float(face["confidence"]),
                    "qualityIssue": quality_info.get("reason", "unknown"),
                })
                boxes.append(new_box)
                names.append("Unknown")
                confidences.append(0)
                continue

            embedding = get_embedding(preprocessed)

            # Match against stored encodings (cosine + SVM ensemble)
            match = _match_face(
                embedding, student_encodings, threshold, enrolled_student_ids
            )

            boxes.append(new_box)

            if match:
                recognized_students.append({
                    "studentId": match["studentId"],
                    "registrationNumber": match.get("registrationNumber", ""),
                    "name": match.get("name", ""),
                    "confidence": float(match["confidence"]),
                    "distance": float(match["distance"]),
                    "matchMethod": match.get("matchMethod", "distance_only"),
                    "faceLocation": {
                        "x1": int(new_box[0]),
                        "y1": int(new_box[1]),
                        "x2": int(new_box[2]),
                        "y2": int(new_box[3]),
                    },
                })
                names.append(match.get("name", match["studentId"][:8]))
                confidences.append(float(match["confidence"]))
                rec_embeddings.append(embedding)
                rec_box_indices.append(len(boxes) - 1)
            else:
                unknown_faces.append({
                    "faceLocation": {
                        "x1": int(new_box[0]),
                        "y1": int(new_box[1]),
                        "x2": int(new_box[2]),
                        "y2": int(new_box[3]),
                    },
                    "confidence": float(face["confidence"]),
                })
                names.append("Unknown")
                confidences.append(0)

        except Exception as e:
            logger.warning(f"Error processing face: {e}")
            unknown_faces.append({
                "faceLocation": {
                    "x1": int(box[0]),
                    "y1": int(box[1]),
                    "x2": int(box[0] + box[2]),
                    "y2": int(box[1] + box[3]),
                },
                "confidence": float(face["confidence"]),
            })
            boxes.append((box[0], box[1], box[0] + box[2], box[1] + box[3]))
            names.append("Unknown")
            confidences.append(0)

    # Deduplicate with re-matching: when multiple faces match the same student,
    # keep the best match and re-match demoted faces against remaining students
    claimed_students: set = set()
    max_dedup_iterations = 10

    for _dedup_round in range(max_dedup_iterations):
        # Build conflict map: student_id -> [(rec_index, distance)]
        student_faces: Dict[str, list] = defaultdict(list)
        for idx, rec in enumerate(recognized_students):
            if rec is not None:
                student_faces[rec["studentId"]].append((idx, rec["distance"]))

        has_conflict = False
        for sid, face_list in student_faces.items():
            if len(face_list) <= 1:
                # No conflict — mark as claimed
                claimed_students.add(sid)
                continue

            has_conflict = True
            # Keep best match (lowest distance), re-match others
            face_list.sort(key=lambda x: x[1])
            best_idx = face_list[0][0]
            claimed_students.add(sid)

            for idx, _dist in face_list[1:]:
                # Try to re-match this face excluding already-claimed students
                emb = rec_embeddings[idx]
                filtered_encodings = {
                    k: v for k, v in student_encodings.items()
                    if k not in claimed_students
                }
                new_match = _match_face(
                    emb, filtered_encodings, threshold, enrolled_student_ids
                )
                box_idx = rec_box_indices[idx]

                if new_match:
                    # Found an alternative match
                    recognized_students[idx] = {
                        "studentId": new_match["studentId"],
                        "registrationNumber": new_match.get("registrationNumber", ""),
                        "name": new_match.get("name", ""),
                        "confidence": float(new_match["confidence"]),
                        "distance": float(new_match["distance"]),
                        "matchMethod": new_match.get("matchMethod", "distance_only") + "_rematch",
                        "faceLocation": recognized_students[idx]["faceLocation"],
                    }
                    names[box_idx] = new_match.get("name", new_match["studentId"][:8])
                    confidences[box_idx] = float(new_match["confidence"])
                    logger.info(
                        f"  Dedup rematch: face re-assigned to {new_match.get('name', '')} "
                        f"(dist={new_match['distance']:.4f})"
                    )
                else:
                    # No alternative — demote to unknown
                    unknown_faces.append({
                        "faceLocation": recognized_students[idx]["faceLocation"],
                        "confidence": recognized_students[idx]["confidence"],
                    })
                    names[box_idx] = "Unknown"
                    confidences[box_idx] = 0
                    recognized_students[idx] = None

        if not has_conflict:
            break

    recognized_students = [r for r in recognized_students if r is not None]

    # Save annotated image if output dir specified
    annotated_path = None
    if output_dir and boxes:
        annotated = draw_annotations(image_rgb, boxes, names, confidences)
        annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        os.makedirs(output_dir, exist_ok=True)
        annotated_path = os.path.join(output_dir, "annotated.jpg")
        cv2.imwrite(annotated_path, annotated_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

    processing_time = int((time.time() - start_time) * 1000)

    # --- Performance & Accuracy Metrics ---
    total_faces = len(faces)
    total_recognized = len(recognized_students)
    total_unknown = len(unknown_faces)
    recognition_rate = (total_recognized / total_faces * 100) if total_faces > 0 else 0

    logger.info("=" * 60)
    logger.info("  RECOGNITION METRICS")
    logger.info("=" * 60)
    logger.info(f"  Threshold:          {threshold}")
    logger.info(f"  Distance metric:    Cosine")
    logger.info(f"  SVM available:      {face_classifier.is_trained}")
    logger.info(f"  Image size:         {image_rgb.shape[1]}x{image_rgb.shape[0]}")
    logger.info(f"  Enrolled students:  {len(enrolled_student_ids)}")
    logger.info(f"  Students w/ encod:  {len(student_encodings)}")
    logger.info("-" * 60)
    logger.info(f"  Faces detected:     {total_faces}")
    logger.info(f"  Faces recognized:   {total_recognized}")
    logger.info(f"  Unknown faces:      {total_unknown}")
    logger.info(f"  Skipped (quality):  {skipped_quality}")
    logger.info(f"  Recognition rate:   {recognition_rate:.1f}%")
    logger.info("-" * 60)

    if recognized_students:
        dists = [s["distance"] for s in recognized_students]
        confs = [s["confidence"] for s in recognized_students]
        logger.info("  Recognized Faces:")
        for s in recognized_students:
            logger.info(
                f"    {s['name']:25s}  dist={s['distance']:.4f}  "
                f"conf={s['confidence']:.1f}%  method={s.get('matchMethod', 'n/a')}"
            )
        logger.info("-" * 60)
        logger.info(f"  Avg distance:       {np.mean(dists):.4f}")
        logger.info(f"  Min distance:       {np.min(dists):.4f}")
        logger.info(f"  Max distance:       {np.max(dists):.4f}")
        logger.info(f"  Avg confidence:     {np.mean(confs):.1f}%")
        logger.info(f"  Min confidence:     {np.min(confs):.1f}%")
        logger.info(f"  Max confidence:     {np.max(confs):.1f}%")

    logger.info(f"  Processing time:    {processing_time}ms")
    logger.info("=" * 60)

    return {
        "facesDetected": total_faces,
        "facesRecognized": total_recognized,
        "recognizedStudents": recognized_students,
        "unknownFaces": unknown_faces,
        "annotatedImagePath": annotated_path,
        "processingTimeMs": processing_time,
        "metrics": {
            "threshold": threshold,
            "distanceMetric": "cosine",
            "svmAvailable": face_classifier.is_trained,
            "imageWidth": int(image_rgb.shape[1]),
            "imageHeight": int(image_rgb.shape[0]),
            "enrolledStudents": len(enrolled_student_ids),
            "studentsWithEncodings": len(student_encodings),
            "facesDetected": total_faces,
            "facesRecognized": total_recognized,
            "unknownFaces": total_unknown,
            "skippedQuality": skipped_quality,
            "recognitionRate": round(recognition_rate, 1),
            "avgDistance": round(float(np.mean(dists)), 4) if recognized_students else None,
            "minDistance": round(float(np.min(dists)), 4) if recognized_students else None,
            "maxDistance": round(float(np.max(dists)), 4) if recognized_students else None,
            "avgConfidence": round(float(np.mean(confs)), 1) if recognized_students else None,
            "processingTimeMs": processing_time,
        },
    }


def _match_face(
    embedding: np.ndarray,
    student_encodings: Dict[str, Dict],
    threshold: float,
    enrolled_ids: Optional[List[str]] = None,
) -> Optional[Dict]:
    """
    Match a face embedding against stored student encodings.
    Uses cosine distance with weighted min/avg approach + SVM ensemble.
    """
    if not student_encodings:
        return None

    best_match = None
    best_combined = float("inf")

    for student_id, data in student_encodings.items():
        encodings = data["encodings"]
        if encodings is None or len(encodings) == 0:
            continue

        # L2-normalize stored encodings on-the-fly
        norms = np.linalg.norm(encodings, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1)
        normed_encodings = encodings / norms

        # Cosine distance: 1 - dot product of unit vectors
        similarities = np.dot(normed_encodings, embedding)
        distances = 1.0 - similarities

        avg_dist = float(np.mean(distances))
        min_dist = float(np.min(distances))

        # Weighted: 70% min, 30% avg
        combined = 0.7 * min_dist + 0.3 * avg_dist

        # Per-student adaptive threshold
        student_threshold = encoding_store.get_student_threshold(
            student_id, threshold
        )

        if combined < best_combined:
            best_combined = combined
            best_match = {
                "studentId": student_id,
                "registrationNumber": data.get("registrationNumber", ""),
                "name": data.get("name", ""),
                "distance": float(combined),
                "_threshold": student_threshold,
            }

    if not best_match:
        return None

    effective_threshold = best_match.pop("_threshold")

    if best_combined <= effective_threshold:
        # Distance-based match found — check SVM ensemble
        # Map cosine distance to confidence:
        #   distance 0.0 → 99%, distance == threshold → 55%
        # This gives a realistic range where good matches (0.1-0.3) score 75-95%
        dist_confidence = max(
            0, min(99, (1 - best_combined / (effective_threshold * 1.8)) * 100)
        )

        svm_result = face_classifier.predict(embedding, enrolled_ids)

        if svm_result and svm_result["studentId"] == best_match["studentId"]:
            # Both agree — boost confidence
            svm_prob = svm_result["probability"]
            final_confidence = 0.6 * dist_confidence + 0.4 * (svm_prob * 100)
            best_match["confidence"] = round(min(99, final_confidence), 1)
            best_match["matchMethod"] = "ensemble_agree"
        elif svm_result and svm_result["probability"] > 0.7:
            # SVM disagrees strongly — reduce confidence
            best_match["confidence"] = round(dist_confidence * 0.75, 1)
            best_match["matchMethod"] = "distance_only_svm_disagree"
        else:
            # SVM not confident or not trained — use distance only
            best_match["confidence"] = round(dist_confidence, 1)
            best_match["matchMethod"] = "distance_only"

        return best_match

    # Near-miss: check if SVM can rescue
    if best_combined <= effective_threshold * 1.15:
        svm_result = face_classifier.predict(embedding, enrolled_ids)
        if (
            svm_result
            and svm_result["studentId"] == best_match["studentId"]
            and svm_result["probability"] > 0.75
        ):
            best_match["confidence"] = round(svm_result["probability"] * 70, 1)
            best_match["matchMethod"] = "svm_rescue"
            return best_match

    return None


def _augment_face(face_image: np.ndarray) -> List[np.ndarray]:
    """Generate augmented versions of a preprocessed face for more diverse encodings."""
    augmented = []

    # Horizontal flip
    augmented.append(cv2.flip(face_image, 1))

    # Slight brightness increase
    augmented.append(cv2.convertScaleAbs(face_image, alpha=1.0, beta=15))

    # Slight brightness decrease
    augmented.append(cv2.convertScaleAbs(face_image, alpha=1.0, beta=-15))

    # Slight Gaussian blur (simulates distance)
    augmented.append(cv2.GaussianBlur(face_image, (3, 3), 0.5))

    return augmented


def generate_encodings_for_images(
    image_paths: List[str],
    augment: bool = True,
) -> List[np.ndarray]:
    """
    Generate face encodings from a list of image paths.
    With augmentation, produces 5 embeddings per successfully detected face.
    """
    embeddings = []

    for image_path in image_paths:
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"Could not read image: {image_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = detect_faces(img_rgb)

            if not faces:
                logger.warning(f"No face detected in: {image_path}")
                continue

            # Use the face with highest confidence
            best_face = max(faces, key=lambda f: f["confidence"])
            preprocessed, _ = preprocess_face(
                img_rgb,
                best_face["box"],
                best_face.get("left_eye"),
                best_face.get("right_eye"),
            )

            # Quality check
            is_acceptable, quality_info = assess_face_quality(preprocessed)
            if not is_acceptable:
                logger.warning(
                    f"Low quality face in {image_path}: {quality_info}, skipping"
                )
                continue

            # Original embedding
            embedding = get_embedding(preprocessed)
            embeddings.append(embedding)

            # Augmented embeddings
            if augment:
                for aug_face in _augment_face(preprocessed):
                    aug_embedding = get_embedding(aug_face)
                    embeddings.append(aug_embedding)

        except Exception as e:
            logger.error(f"Error generating encoding for {image_path}: {e}")

    return embeddings
