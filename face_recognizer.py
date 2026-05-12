"""
face_recognizer.py — RetinaFace detection + ArcFace 512-D encoding via insightface.

Pipeline:
  1. RetinaFace (ONNX) — face detection, handles small/angled/occluded faces
  2. ArcFace (mobilefacenet, 512-D) — L2-normalized embedding with cosine similarity

Uses the insightface package (buffalo_s model set), auto-downloads models.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ArcFace cosine similarity threshold
DEFAULT_THRESHOLD = 0.6


class FaceRecognizer:
    """
    RetinaFace detection + ArcFace 512-D recognition via insightface.

    Public API:
      detect_faces(image) -> List[dict]  (each has 'bbox', 'embedding', 'det_score')
      get_face_encoding(image) -> (512-D ndarray, face_dict) | None
      register_face(images) -> 512-D averaged ndarray | None
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        self.threshold = threshold
        self._app = None
        self._init_model()

    def _init_model(self):
        """
        Lazy-init models — retries once if first load fails (e.g. download timeout).
        insightface auto-downloads buffalo_s ONNX models on first run.
        """
        for attempt in range(2):
            try:
                from insightface.app import FaceAnalysis
                self._app = FaceAnalysis(
                    name="buffalo_s",
                    providers=["CPUExecutionProvider"],
                )
                self._app.prepare(ctx_id=0, det_size=(640, 480))
                log.info(
                    "FaceRecognizer: RetinaFace + ArcFace loaded (buffalo_s)"
                )
                return
            except Exception as e:
                log.warning(
                    "FaceRecognizer init attempt %d/2 failed: %s",
                    attempt + 1, e,
                )
                import time
                time.sleep(1.0)

        log.error("FaceRecognizer: failed to initialize after 2 attempts")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_faces(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all faces in image using RetinaFace.

        Returns:
            List of dicts with keys:
              - 'bbox': [x1, y1, x2, y2] (int pixel coords)
              - 'embedding': 512-D L2-normalized numpy array
              - 'det_score': float detection confidence
              - 'kps': (optional) 5x2 landmark array
        """
        if self._app is None:
            return []

        raw_faces = self._app.get(image)
        results = []
        for face in raw_faces:
            bbox = face.bbox.astype(int).tolist()  # [x1, y1, x2, y2]
            embedding = face.normed_embedding.astype(np.float32)
            results.append({
                'bbox': bbox,
                'embedding': embedding,
                'det_score': float(face.det_score),
                'kps': face.kps.tolist() if hasattr(face, 'kps') and face.kps is not None else None,
            })

        # Sort by detection confidence (highest first)
        results.sort(key=lambda f: f['det_score'], reverse=True)
        return results

    def get_face_encoding(
        self, image: np.ndarray
    ) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Detect largest face and return (512-D embedding, face_dict).

        Returns None if no face detected.
        """
        faces = self.detect_faces(image)
        if not faces:
            return None

        # Largest by bbox area
        largest = max(
            faces,
            key=lambda f: (f['bbox'][2] - f['bbox'][0]) *
                          (f['bbox'][3] - f['bbox'][1]),
        )
        return largest['embedding'], largest

    def register_face(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Average ArcFace embeddings across multiple images.

        Args:
            images: List of BGR numpy arrays.

        Returns:
            Averaged 512-D L2-normalized embedding, or None if no face found.
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
        return avg.astype(np.float32)

    @staticmethod
    def bbox_to_rect(bbox):
        """
        Convert bbox list [x1, y1, x2, y2] to tuple (x1, y1, x2, y2).
        Handles both list and numpy array.
        """
        return (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
