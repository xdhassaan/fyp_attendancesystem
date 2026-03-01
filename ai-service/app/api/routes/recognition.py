"""
Face recognition endpoint.
Accepts a class photo and returns recognized students.
"""

import os
import json
import base64
import logging
import tempfile
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.services.model_loader import model_loader
from app.services.recognition_service import recognize_faces_in_image

logger = logging.getLogger(__name__)
router = APIRouter()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@router.post("/recognize")
async def recognize_faces(
    image: UploadFile = File(...),
    student_ids: str = Form("[]"),
    threshold: float = Form(0.6),
):
    """
    Process a class photo for face recognition.

    - **image**: Class photo (JPEG/PNG)
    - **student_ids**: JSON array of enrolled student IDs
    - **threshold**: Recognition threshold (0.3-1.0, default 0.6)
    """
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Models are still loading")

    # Validate file type
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Use JPEG, PNG, or WebP.")

    # Parse student IDs
    try:
        enrolled_ids = json.loads(student_ids)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid student_ids format")

    # Read image
    contents = await image.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=422, detail="Could not decode image")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Determine output directory for annotated image
    output_dir = os.path.join(BASE_DIR, "output", "annotated")

    # Run recognition
    result = recognize_faces_in_image(
        img_rgb,
        enrolled_student_ids=enrolled_ids,
        threshold=threshold,
        output_dir=output_dir,
    )

    if result["facesDetected"] == 0:
        raise HTTPException(status_code=422, detail="No faces detected in the image")

    # Include annotated image as base64 so frontend can display it
    annotated_path = result.get("annotatedImagePath")
    if annotated_path and os.path.exists(annotated_path):
        with open(annotated_path, "rb") as f:
            result["annotatedImageBase64"] = base64.b64encode(f.read()).decode("utf-8")

    return result
