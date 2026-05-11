"""
liveness_detector.py — Eye-blink-based liveness detection via dlib 68-point landmarks.

EAR (Eye Aspect Ratio) from the Drowsiness project pattern adapted for dlib.
Liveness passes if >= 2 natural blinks detected within observation window.
A blink = EAR drops below threshold then recovers (open→close→open cycle).
Photos/videos won't produce natural blink patterns.

Public interface:
  check_liveness(frames) -> {"passed": bool, "score": float, ...}
"""

import os
import sys
from typing import Dict, List, Optional

import cv2
import dlib
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────
EAR_THRESHOLD = 0.22          # below this = eyes closed
BLINK_FRAMES_MIN = 2          # min consecutive closed frames to count as blink
BLINK_FRAMES_MAX = 10         # max consecutive closed frames (beyond = just squinting)
BLINKS_REQUIRED = 2           # blinks needed to pass liveness
OBSERVATION_FRAMES = 30       # frames to observe (~2s at 15fps)

EAR_SMOOTH_ALPHA = 0.3

# dlib 68-point landmarks: left eye = 36-41, right eye = 42-47
LEFT_EYE_IDX  = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))

SHAPE_PREDICTOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "shape_predictor_68_face_landmarks.dat",
)


# ── Helpers ────────────────────────────────────────────────────────────

def eye_aspect_ratio(eye_pts: np.ndarray) -> float:
    """
    Compute EAR for a single eye from 6 landmark points.

    Args:
        eye_pts: (6, 2) array of (x, y) coordinates.
    Returns:
        EAR value.
    """
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3])
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0


# ── LivenessDetector ───────────────────────────────────────────────────

class LivenessDetector:
    """EAR-based blink detector using dlib facial landmarks."""

    def __init__(self):
        self._predictor: Optional[dlib.shape_predictor] = None
        self._detector: Optional[dlib.frone_face_detector] = None
        self._frame_count = 0

    def _ensure_models(self):
        if self._predictor is not None:
            return

        # Check shape predictor exists
        if not os.path.exists(SHAPE_PREDICTOR_PATH):
            print(f"[Liveness] shape_predictor_68_face_landmarks.dat not found at:")
            print(f"          {SHAPE_PREDICTOR_PATH}")
            print("          Download from:")
            print("          http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            raise FileNotFoundError(f"Missing: {SHAPE_PREDICTOR_PATH}")

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)

    def check_liveness(self, frames: List[np.ndarray]) -> Dict:
        """
        Check liveness by detecting natural blinks across frames.

        Args:
            frames: List of BGR numpy arrays (OpenCV default).

        Returns:
            dict with keys: passed, score, blinks_detected, details.
        """
        self._ensure_models()

        smooth_ear = 0.3
        closed_frames = 0
        blink_count = 0
        was_closed = False
        total_frames = min(len(frames), OBSERVATION_FRAMES)

        for frame in frames[:total_frames]:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self._detector(gray, 0)

            if not faces:
                smooth_ear = 0.3
                closed_frames = 0
                was_closed = False
                continue

            # Use largest face
            face = max(faces, key=lambda r: r.width() * r.height())
            shape = self._predictor(gray, face)

            # Extract eye landmarks
            left_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in LEFT_EYE_IDX], dtype=np.float32)
            right_pts = np.array([(shape.part(i).x, shape.part(i).y) for i in RIGHT_EYE_IDX], dtype=np.float32)

            left_ear = eye_aspect_ratio(left_pts)
            right_ear = eye_aspect_ratio(right_pts)
            raw_ear = (left_ear + right_ear) / 2.0
            smooth_ear = EAR_SMOOTH_ALPHA * raw_ear + (1 - EAR_SMOOTH_ALPHA) * smooth_ear
            ear = smooth_ear

            is_closed = ear < EAR_THRESHOLD

            if is_closed:
                closed_frames += 1
                was_closed = True
            else:
                if was_closed and BLINK_FRAMES_MIN <= closed_frames <= BLINK_FRAMES_MAX:
                    blink_count += 1
                closed_frames = 0
                was_closed = False

        # Check end-of-sequence
        if was_closed and BLINK_FRAMES_MIN <= closed_frames <= BLINK_FRAMES_MAX:
            blink_count += 1

        passed = blink_count >= BLINKS_REQUIRED
        score = min(blink_count / BLINKS_REQUIRED, 1.0)

        return {
            'passed': passed,
            'score': score,
            'blinks_detected': blink_count,
            'blinks_required': BLINKS_REQUIRED,
            'details': f'{blink_count}/{BLINKS_REQUIRED} blinks detected',
        }
