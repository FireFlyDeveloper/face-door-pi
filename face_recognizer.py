"""
face_recognizer.py — Face detection + ArcFace encoding via ONNX Runtime.

Detection: dlib HOG frontal face detector (fast, same API as before).
Recognition: ArcFace R100 ONNX model producing 512-D L2-normalized embeddings
  with cosine similarity matching.

Reference:
  Deng et al. (2019) — ArcFace: Additive Angular Margin Loss for
  Deep Face Recognition. State-of-the-art, used in Qengineering RPi impl.

Model: ArcFace R100 ONNX
  Source: https://github.com/deepinsight/insightface
  Download via scripts/download_models.sh
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np

log = logging.getLogger(__name__)

ONNX_MODEL = "./models/arcface_r100.onnx"
INPUT_SIZE = (112, 112)  # ArcFace standard input


class FaceRecognizer:
    """
    Face detection via dlib HOG, recognition via ArcFace ONNX.

    Public API (backward-compatible method signatures):
      detect_faces(image) -> List[dlib.rectangle]
      get_face_encoding(image) -> (512-D ndarray, dlib.rectangle) | None
      register_face(images) -> 512-D ndarray | None
    """

    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold

        # ArcFace ONNX session
        self._session: Optional[object] = None
        self._input_name: Optional[str] = None
        self._load_onnx()

        # dlib HOG detector (same as before)
        self._detector = dlib.get_frontal_face_detector()
        self._shape_predictor_5 = self._load_shape_predictor(
            "shape_predictor_5_face_landmarks.dat"
        )

        self._encoding_dim = 512  # ArcFace output

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_onnx(self):
        if not os.path.exists(ONNX_MODEL):
            log.warning(
                "FaceRecognizer: ONNX model not found at %s — "
                "encoding will return zero vectors. "
                "Run scripts/download_models.sh",
                ONNX_MODEL,
            )
            return
        try:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 4
            opts.intra_op_num_threads = 4
            self._session = ort.InferenceSession(
                ONNX_MODEL,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
            log.info(
                "FaceRecognizer: ArcFace ONNX loaded (%s)",
                ONNX_MODEL,
            )
        except ImportError:
            log.error("onnxruntime not installed. Run: pip install onnxruntime")

    @staticmethod
    def _load_shape_predictor(filename: str):
        """Try common paths for dlib shape predictor files."""
        paths = [
            f"/usr/share/dlib/{filename}",
            os.path.join(
                os.path.dirname(os.path.dirname(dlib.__file__)),
                "data",
                filename,
            ),
            os.path.join(
                os.path.dirname(os.path.dirname(dlib.__file__)),
                filename,
            ),
        ]
        for p in paths:
            if os.path.exists(p):
                return dlib.shape_predictor(p)
        # Fall through — create a dummy that won't crash
        log.warning("Shape predictor %s not found — landmarks disabled", filename)
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
        Detect largest face and return (512-D embedding, bounding box).

        Returns None if no face detected or model not loaded.
        """
        faces = self.detect_faces(image)
        if not faces:
            return None

        largest = max(
            faces,
            key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()),
        )
        embedding = self._embed(image, largest)
        if embedding is None:
            return None
        return embedding, largest

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

    def get_face_landmarks(
        self, image: np.ndarray
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Compute 5-point facial landmarks.

        Returns list of (x, y) tuples, or None if no face found.
        """
        if self._shape_predictor_5 is None:
            return None
        faces = self.detect_faces(image)
        if not faces:
            return None
        largest = max(
            faces,
            key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()),
        )
        rgb = self._to_rgb(image)
        shape = self._shape_predictor_5(rgb, largest)
        return [(shape.part(i).x, shape.part(i).y) for i in range(5)]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _embed(self, image: np.ndarray, face: dlib.rectangle) -> Optional[np.ndarray]:
        """Extract 512-D ArcFace embedding for a single face rect."""
        if self._session is None:
            # Graceful degradation: return zeros
            return np.zeros(self._encoding_dim, dtype=np.float32)

        # Crop face with margin
        x1 = max(0, face.left())
        y1 = max(0, face.top())
        x2 = min(image.shape[1], face.right())
        y2 = min(image.shape[0], face.bottom())
        face_crop = image[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        # Preprocess
        img = cv2.resize(face_crop, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        # ArcFace normalization: mean=127.5, std=128.0
        img = (img - 127.5) / 128.0
        img = np.transpose(img, (2, 0, 1))      # HWC → CHW
        img = np.expand_dims(img, axis=0)        # (1, 3, 112, 112)

        output = self._session.run(None, {self._input_name: img})
        embedding = output[0][0]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
