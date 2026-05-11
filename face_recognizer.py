"""
face_recognizer.py — Face detection + encoding using face_recognition / dlib.

Provides:
  - Face detection via dlib's HOG + CNN frontal face detector
  - 128-D face encoding via dlib's ResNet face recognition model
  - Landmark detection (68-point) for liveness / alignment
  - Recognition against a known-face dictionary with Euclidean distance
"""

from typing import Any, Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np


class FaceRecognizer:
    """Face detection, encoding, landmark extraction, and recognition."""

    def __init__(self):
        # HOG-based face detector (fast, default)
        self._detector = dlib.get_frontal_face_detector()

        # 5-point landmark predictor — sufficient for face alignment
        # We also load the 68-point model for liveness detection (EAR)
        predictor_5_path = "/usr/share/dlib/shape_predictor_5_face_landmarks.dat"
        predictor_68_path = "/usr/share/dlib/shape_predictor_68_face_landmarks.dat"

        # Fall back to searching common paths if not found
        import os

        for p in [predictor_5_path, predictor_68_path]:
            if not os.path.exists(p):
                # Try alternate paths on Raspberry Pi / Debian
                alt = os.path.join(
                    os.path.dirname(os.path.dirname(dlib.__file__)),
                    "data",
                    os.path.basename(p),
                )
                if os.path.exists(alt):
                    if "68" in p:
                        predictor_68_path = alt
                    else:
                        predictor_5_path = alt

        self._shape_predictor_5 = dlib.shape_predictor(predictor_5_path)
        self._shape_predictor_68 = dlib.shape_predictor(predictor_68_path)

        # Face recognition model (ResNet 128-D embedding)
        self._face_rec_model = dlib.face_recognition_model_v1(
            "/usr/share/dlib/dlib_face_recognition_resnet_model_v1.dat"
        )

        self._threshold = 0.6  # Euclidean distance threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_faces(self, image: np.ndarray) -> List[dlib.rectangle]:
        """
        Detect all face bounding boxes in an image.

        Args:
            image: BGR numpy array (what OpenCV/cv2 reads).

        Returns:
            List of dlib.rectangle objects (left, top, right, bottom).
        """
        # dlib expects RGB; convert if needed
        rgb = self._to_rgb(image)
        # No upsampling — faster for close-range door use
        # (person will be ~0.5-1m from camera)
        return self._detector(rgb, 0)

    def get_face_encoding(
        self, image: np.ndarray
    ) -> Optional[Tuple[np.ndarray, dlib.rectangle]]:
        """
        Detect the largest face in the image and return its 128-D encoding
        and bounding box.

        Args:
            image: BGR numpy array.

        Returns:
            (encoding, face_location) if a face is found, else None.
            encoding is a 128-D np.ndarray of float64.
        """
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return None

        # Pick the largest face by area
        largest = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))

        rgb = self._to_rgb(image)
        shape = self._shape_predictor_5(rgb, largest)
        encoding = np.array(
            self._face_rec_model.compute_face_descriptor(rgb, shape), dtype=np.float64
        )
        return encoding, largest

    def register_face(self, images: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        Register a face from a list of images (typically 10). Detects a face
        in each valid image, computes encodings, and averages them.

        Args:
            images: List of BGR numpy arrays (at least 1, ideally 10).

        Returns:
            Averaged 128-D encoding as np.ndarray, or None if no face found
            in any image.
        """
        valid_encodings: List[np.ndarray] = []
        for i, img in enumerate(images):
            try:
                result = self.get_face_encoding(img)
                if result is not None:
                    encoding, _ = result
                    valid_encodings.append(encoding)
            except Exception:
                continue

        if len(valid_encodings) == 0:
            return None

        # Average all valid encodings
        avg_encoding = np.mean(valid_encodings, axis=0)
        # Renormalize to unit length
        avg_encoding = avg_encoding / np.linalg.norm(avg_encoding)
        return avg_encoding

    def recognize(
        self,
        face_encoding: np.ndarray,
        known_faces: Dict[str, Dict[str, Any]],
    ) -> Optional[Tuple[str, float]]:
        """
        Match a face encoding against a dictionary of known faces.

        Args:
            face_encoding: 128-D query encoding.
            known_faces: Dict of face_id -> {'encoding': [10 x np.ndarray], ...}.
                The first encoding from the stored list is used as the reference
                (the averaged registration encoding is typically at index 0,
                or all 10 are the same averaged encoding from registration).

        Returns:
            (best_match_id, smallest_distance) if within threshold, else None.
        """
        best_id: Optional[str] = None
        best_dist: float = float("inf")

        for face_id, face_data in known_faces.items():
            encodings = face_data.get("encoding", [])
            for stored_enc in encodings:
                dist = np.linalg.norm(face_encoding - stored_enc)
                if dist < best_dist:
                    best_dist = dist
                    best_id = face_id

        if best_id is not None and best_dist <= self._threshold:
            return best_id, best_dist
        return None

    def get_face_landmarks(
        self, image: np.ndarray
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Compute 68-point facial landmarks (for liveness detection).

        Args:
            image: BGR numpy array.

        Returns:
            List of (x, y) tuples for the 68 landmarks, or None if no face.
        """
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return None

        largest = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
        rgb = self._to_rgb(image)
        shape = self._shape_predictor_68(rgb, largest)

        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        return landmarks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_rgb(image: np.ndarray) -> np.ndarray:
        """Convert grayscale or BGR to 3-channel RGB for dlib."""
        if image.ndim == 2:
            # Grayscale (H x W) → stack to 3-channel RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
