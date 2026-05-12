"""
ir_liveness.py — Video-based anti-fool liveness via face motion analysis.

Replaced NoIR spectral analysis (no IR LED available) with multi-frame
motion-based detection. A live person exhibits natural micro-movements
(breathing, posture sway), while a static photo/screen shows near-zero
variance in face position and pixel content across frames.

Principle:
  Track face bounding box center (cx, cy) and area across N frames.
  A live face will show measurable variance due to natural motion;
  a printed photo or phone screen held still will be nearly static.

Metrics (computed across all frames in a collection window):
  1. Position variance: std(cx) + std(cy) — live > threshold
  2. Area variance: std(area) / mean(area) — normalised size fluctuation
  3. Pixel variance: mean absolute diff between consecutive face crops

Public interface:
  IRLivenessDetector.check_liveness(frames: list, face_rects: list) -> dict
  IRLivenessDetector.reset()
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Thresholds (tune empirically) ───────────────────────────────────────
# Total positional jitter needed to pass (std_x + std_y in pixels)
POSITION_VARIANCE_MIN = 2.0   # sum of std(cx) + std(cy)
# Minimum number of frames required for a valid check
MIN_FRAMES = 10
# Face rect area as percentage of total frame — below this is too small
MIN_FACE_AREA_RATIO = 0.01     # 1% of 640x480 = ~3072px

# Debug: print per-frame metrics
_DEBUG = False


class IRLivenessDetector:
    """Video-based anti-fool liveness via face motion analysis.

    Collects a sequence of face detections across multiple frames and
    analyzes positional variance to distinguish live persons from static
    photos or screens.

    Usage:
        detector = IRLivenessDetector()
        detector.reset()
        # ... collect frames in a loop ...
        result = detector.check_liveness(frames, face_rects)
    """

    def __init__(self, debug: bool = False):
        global _DEBUG
        _DEBUG = debug
        self.reset()

    def reset(self):
        """Clear state between liveness checks."""
        self._frame_count = 0
        self._centers: List[Tuple[float, float]] = []  # (cx, cy)
        self._areas: List[float] = []                   # bounding box area
        self._face_crops: List[np.ndarray] = []         # aligned face crops

    def add_frame(self, frame: np.ndarray, face_rect) -> bool:
        """Process one frame during collection.

        Args:
            frame: BGR numpy array.
            face_rect: dlib rectangle of the detected face.

        Returns:
            True if the frame was valid and added, False otherwise.
        """
        h, w = frame.shape[:2]
        x1 = max(face_rect.left(), 0)
        y1 = max(face_rect.top(), 0)
        x2 = min(face_rect.right(), w)
        y2 = min(face_rect.bottom(), h)

        if x2 <= x1 or y2 <= y1:
            return False

        face_w = x2 - x1
        face_h = y2 - y1
        area = face_w * face_h

        # Reject tiny detections
        if area < MIN_FACE_AREA_RATIO * w * h:
            return False

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        self._centers.append((cx, cy))
        self._areas.append(float(area))
        self._face_crops.append(frame[y1:y2, x1:x2])
        self._frame_count += 1
        return True

    def check_liveness(
        self,
        frames: Optional[List[np.ndarray]] = None,
        face_rects: Optional[List] = None,
    ) -> Dict:
        """Analyze collected frames for face-motion liveness.

        If frames and face_rects are provided, they are processed
        immediately (batch mode). Otherwise, uses the accumulated
        frames from add_frame().

        Args:
            frames: Optional list of BGR frames.
            face_rects: Optional list of dlib rectangles (same length).

        Returns:
            dict with: passed, score, metrics, details.
        """
        # Batch mode: process all at once
        if frames is not None and face_rects is not None:
            self.reset()
            for f, r in zip(frames, face_rects):
                if r is not None:
                    self.add_frame(f, r)

        if self._frame_count < MIN_FRAMES:
            return {
                'passed': False,
                'score': 0.0,
                'details': f'insufficient frames ({self._frame_count} < {MIN_FRAMES})',
                'blinks_detected': 0,
                'position_variance': 0.0,
                'area_variance': 0.0,
                'pixel_variance': 0.0,
            }

        # ── Metric 1: Position Variance ──────────────────────────
        cx_arr = np.array([c[0] for c in self._centers])
        cy_arr = np.array([c[1] for c in self._centers])
        pos_variance = float(np.std(cx_arr) + np.std(cy_arr))

        # ── Metric 2: Area Variance (normalised) ─────────────────
        area_arr = np.array(self._areas)
        mean_area = float(np.mean(area_arr))
        area_variance = float(np.std(area_arr) / mean_area) if mean_area > 0 else 0.0

        # ── Metric 3: Pixel Variance (frame-to-frame diff) ───────
        pixel_variance = 0.0
        if len(self._face_crops) >= 3:
            diffs = []
            for i in range(1, len(self._face_crops)):
                # Resize to uniform size for comparison
                prev = cv2.resize(self._face_crops[i - 1], (100, 100))
                curr = cv2.resize(self._face_crops[i], (100, 100))
                diff = cv2.absdiff(prev, curr)
                diffs.append(float(np.mean(diff)))
            pixel_variance = float(np.mean(diffs)) if diffs else 0.0

        # ── Decision ─────────────────────────────────────────────
        passed = pos_variance >= POSITION_VARIANCE_MIN

        # Combined score (0-1)
        pos_score = min(1.0, pos_variance / (POSITION_VARIANCE_MIN * 3))
        area_score = min(1.0, area_variance * 50)  # normalised boost
        pix_score = min(1.0, pixel_variance / 10.0)
        combined = (pos_score * 0.6) + (area_score * 0.2) + (pix_score * 0.2)

        details = (
            f"frames={self._frame_count} "
            f"pos_var={pos_variance:.2f}px "
            f"(thresh>={POSITION_VARIANCE_MIN}), "
            f"area_var={area_variance:.4f}, "
            f"pix_var={pixel_variance:.2f}"
        )

        if _DEBUG:
            print(f"[Liveness] {details}")

        # Log individual frame centers for diagnostics
        center_log = ", ".join(
            f"({c[0]:.0f},{c[1]:.0f})" for c in self._centers
        )
        print(f"[Liveness] Centers: [{center_log}]")
        print(f"[Liveness] Position variance: {pos_variance:.2f}px "
              f"{'✅ PASS' if passed else '❌ FAIL'} "
              f"(threshold >= {POSITION_VARIANCE_MIN})")

        return {
            'passed': passed,
            'score': round(combined, 3),
            'position_variance': round(pos_variance, 2),
            'area_variance': round(area_variance, 4),
            'pixel_variance': round(pixel_variance, 2),
            'details': details,
        }
