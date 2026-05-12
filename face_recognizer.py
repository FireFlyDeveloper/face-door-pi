"""
face_recognizer.py — Face detection + encoding using dlib.

Detection: dlib HOG frontal face detector.
Recognition: dlib ResNet 128-D embedding (Euclidean distance matching).

Compatible with existing faces.json (128-D encodings).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np

log = logging.getLogger(__name__)


class FaceRecognizer:
    """
    Face detection via dlib HOG, encoding via dlib ResNet 128-D.
    """

    def __init__(self, threshold: float = 0.6):
        self._threshold = threshold

        # dlib HOG face detector (fast)
        self._detector = dlib.get_frontal_face_detector()

        # Shape predictors for alignment
        self._shape_predictor_5 = self._load_shape_predictor(
            "shape_predictor_5_face_landmarks.dat"
        )
        self._shape_predictor_68 = self._load_shape_predictor(
            "shape_predictor_68_face_landmarks.dat"
        )

        # ResNet face recognition model (128-D embedding)
        self._face_rec_model = self._load_recognition_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_shape_predictor(filename: str):
        """Try common paths for dlib shape predictor files."""
        paths = [
            f"/usr/share/dlib/{filename}",
            os.path.join(
                os.path.dirname(os.path.dirname(dlib.__file__)), "data", filename
            ),
            os.path.join(os.path.dirname(os.path.dirname(dlib.__file__)), filename),
        ]
        for p in paths:
            if os.path.exists(p):
                return dlib.shape_predictor(p)
        log.warning("Shape predictor %s not found — landmarks disabled", filename)
        return None

    @staticmethod
    def _load_recognition_model():
        """Find and load dlib face recognition ResNet model."""
        model_name = "dlib_face_recognition_resnet_model_v1.dat"
        paths = [
            f"/usr/share/dlib/{model_name}",
            os.path.join(
                os.path.dirname(os.path.dirname(dlib.__file__)), "data", model_name
            ),
            os.path.join(os.path.dirname(os.path.dirname(dlib.__file__)), model_name),
        ]
        for p in paths:
            if os.path.exists(p):
                log.info("FaceRecognizer: dlib ResNet model loaded from %s", p)
                return dlib.face_recognition_model_v1(p)
        log.warning("FaceRecognizer: dlib ResNet model not found!")
        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_faces(self, image: np.ndarray) -> List[dlib.rectangle]:
        """
        Detect all face bounding boxes using dlib HOG.

        Returns list of dlib.rectangle (left, top, right, bottom).
        """
        rgb = self._to_rgb(image)
        return self._detector(rgb, 0)  # no upsampling for speed

    def get_face_encoding(
        self, image: np.ndarray
    ) -> Optional[Tuple[np.ndarray, dlib.rectangle]]:
        """
        Detect largest face and return (128-D encoding, bounding box).

        Returns None if no face detected or model not loaded.
        """
        if self._face_rec_model is None:
            return None

        faces = self.detect_faces(image)
        if not faces:
            return None

        largest = max(
            faces,
            key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()),
        )

        rgb = self._to_rgb(image)

        # Use 5-point predictor for alignment
        shape = self._shape_predictor_5(rgb, largest)
        encoding = np.array(
            self._face_rec_model.compute_face_descriptor(rgb, shape),
            dtype=np.float64,
        )
        return encoding, largest

    def register_face(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Average dlib ResNet 128-D encodings across multiple images.

        Args:
            images: List of BGR numpy arrays.

        Returns:
            Averaged 128-D L2-normalized encoding, or None if no face found.
        """
        valid: List[np.ndarray] = []
        for img in images:
            try:
                result = self.get_face_encoding(img)
                if result is not None:
                    enc, _ = result
                    valid.append(enc)
            except Exception:
                continue

        if not valid:
            return None

        avg = np.mean(valid, axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-6)
        return avg.astype(np.float64)

    def get_face_landmarks(
        self, image: np.ndarray
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Compute 68-point facial landmarks.

        Returns list of (x, y) tuples, or None if no face found.
        """
        if self._shape_predictor_68 is None:
            return None
        faces = self.detect_faces(image)
        if not faces:
            return None
        largest = max(
            faces,
            key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()),
        )
        rgb = self._to_rgb(image)
        shape = self._shape_predictor_68(rgb, largest)
        return [(shape.part(i).x, shape.part(i).y) for i in range(68)]

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
