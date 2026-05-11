"""
liveness_detector.py — Real-time eye-blink liveness via dlib EAR.

Two modes:
  check_liveness(frames) — batch mode (legacy)
  process_frame(frame)   — per-frame mode for real-time blink tracking
  reset()                — clear internal state between sessions

EAR (Eye Aspect Ratio) from the Drowsiness project pattern.
Liveness passes if >= 2 natural blinks detected.
A blink = EAR drops below threshold then recovers (open→close→open cycle).
"""

import os
from typing import Dict, List, Optional

import cv2
import dlib
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.22
BLINK_FRAMES_MIN = 1           # single frame drop counts (blinks are fast at 15fps)
BLINK_FRAMES_MAX = 10
BLINKS_REQUIRED = 2
EAR_SMOOTH_ALPHA = 0.3

LEFT_EYE_IDX  = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

SHAPE_PREDICTOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "shape_predictor_68_face_landmarks.dat",
)


def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0


class LivenessDetector:
    """Real-time EAR-based blink detector using dlib."""

    def __init__(self):
        self._predictor: Optional[dlib.shape_predictor] = None
        self._detector: Optional[dlib.frone_face_detector] = None
        self.reset()

    def _ensure_models(self):
        if self._predictor is not None:
            return
        if not os.path.exists(SHAPE_PREDICTOR_PATH):
            raise FileNotFoundError(f"Missing: {SHAPE_PREDICTOR_PATH}")
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    def reset(self):
        """Reset blink tracking state. Call when starting a new observation."""
        self._smooth_ear = 0.3
        self._closed_frames = 0
        self._blink_count = 0
        self._was_closed = False

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame for blink detection.

        Args:
            frame: BGR numpy array.

        Returns:
            dict with: passed, blinks_detected, ear, face_detected.
        """
        self._ensure_models()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector(gray, 0)

        if not faces:
            self._smooth_ear = 0.3
            self._closed_frames = 0
            self._was_closed = False
            return {
                'passed': self._blink_count >= BLINKS_REQUIRED,
                'blinks_detected': self._blink_count,
                'ear': 0.3,
                'face_detected': False,
            }

        face = max(faces, key=lambda r: r.width() * r.height())
        shape = self._predictor(gray, face)

        left_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in LEFT_EYE_IDX], dtype=np.float32)
        right_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in RIGHT_EYE_IDX], dtype=np.float32)

        left_ear = eye_aspect_ratio(left_pts)
        right_ear = eye_aspect_ratio(right_pts)
        raw_ear = (left_ear + right_ear) / 2.0
        self._smooth_ear = EAR_SMOOTH_ALPHA * raw_ear + (1 - EAR_SMOOTH_ALPHA) * self._smooth_ear

        # Use RAW ear for blink detection (smoothing masks brief blinks at low FPS)
        is_closed = raw_ear < EAR_THRESHOLD

        if is_closed:
            self._closed_frames += 1
            self._was_closed = True
        else:
            if self._was_closed and BLINK_FRAMES_MIN <= self._closed_frames <= BLINK_FRAMES_MAX:
                self._blink_count += 1
            self._closed_frames = 0
            self._was_closed = False

        return {
            'passed': self._blink_count >= BLINKS_REQUIRED,
            'blinks_detected': self._blink_count,
            'ear': raw_ear,
            'face_detected': True,
        }

    def check_liveness(self, frames: List[np.ndarray]) -> Dict:
        """Batch mode — processes all frames, returns result."""
        self.reset()
        for frame in frames:
            self.process_frame(frame)
        # Check end-of-sequence
        if self._was_closed and BLINK_FRAMES_MIN <= self._closed_frames <= BLINK_FRAMES_MAX:
            self._blink_count += 1
        passed = self._blink_count >= BLINKS_REQUIRED
        score = min(self._blink_count / BLINKS_REQUIRED, 1.0)
        return {
            'passed': passed,
            'score': score,
            'blinks_detected': self._blink_count,
            'blinks_required': BLINKS_REQUIRED,
            'details': f'{self._blink_count}/{BLINKS_REQUIRED} blinks',
        }
