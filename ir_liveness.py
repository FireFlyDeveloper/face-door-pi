"""
ir_liveness.py — Single-frame anti-fool liveness via texture/edge analysis.

Replaced motion-based approach (too slow, required movement) with a
single-frame passive texture analysis. Distinguishes live faces from
printed photos and screen replays by analyzing:

1. Texture variance (Laplacian): Real skin has fine details (pores, hair)
   producing high-frequency content. Printed paper is flat/smooth.
2. Edge sharpness (Sobel): Real face edges are natural. Screen replays
   have pixel grid artifacts and oversharpened edges.
3. Color saturation spread: Real skin has natural color variation.
   Printed/screen reproductions have compressed or artificial gamut.

All metrics computed on a single frame — zero delay, no user movement.

Public interface:
  IRLivenessDetector.check_liveness(frame, face_rect) -> dict
  IRLivenessDetector.reset()
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Optional, Tuple


# ── Thresholds (tune empirically for Pi NoIR cam 640x480) ──────────────
TEXTURE_VARIANCE_MIN = 2.5     # Laplacian variance — live skin > printed (v2 cam @640x480: ~3.6)
EDGE_STRENGTH_MIN = 4.0        # Mean Sobel magnitude (v2 cam @640x480: ~5.4)
MIN_FACE_REGION = 100          # Minimum face crop pixels


_DEBUG = False


def _face_region(bgr: np.ndarray, face_rect) -> Optional[np.ndarray]:
    """Extract the face region from a BGR frame given a dlib rect."""
    h, w = bgr.shape[:2]
    x1 = max(face_rect.left(), 0)
    y1 = max(face_rect.top(), 0)
    x2 = min(face_rect.right(), w)
    y2 = min(face_rect.bottom(), h)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = bgr[y1:y2, x1:x2]
    if crop.size < MIN_FACE_REGION:
        return None

    return crop


class IRLivenessDetector:
    """Single-frame texture/edge anti-fool liveness detector.

    Uses texture variance analysis to distinguish live faces from
    printed photos or screen replays in a single frame.
    """

    def __init__(self, debug: bool = False):
        global _DEBUG
        _DEBUG = debug
        self.reset()

    def reset(self):
        """Clear state between checks."""
        pass

    def check_liveness(
        self,
        frame: np.ndarray,
        face_rect,
    ) -> Dict:
        """Analyze a single BGR frame for texture-based liveness.

        Args:
            frame: BGR numpy array.
            face_rect: dlib rectangle of the detected face.

        Returns:
            dict with: passed, score, metrics, details.
        """
        crop = _face_region(frame, face_rect)
        if crop is None:
            return {
                'passed': False,
                'score': 0.0,
                'texture_variance': 0.0,
                'edge_strength': 0.0,
                'color_spread': 0.0,
                'details': 'face region too small',
            }

        # ── Metric 1: Texture Variance (Laplacian) ────────────────
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = float(np.var(laplacian))

        # ── Metric 2: Edge Sharpness (Sobel magnitude) ────────────
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_strength = float(np.mean(sobel_mag))

        # ── Metric 3: Color Saturation Spread ──────────────────────
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        color_spread = float(np.std(saturation))

        # ── Decision ───────────────────────────────────────────────
        passed_texture = texture_variance >= TEXTURE_VARIANCE_MIN
        passed_edges = edge_strength >= EDGE_STRENGTH_MIN

        # Pass requires texture variance AND either edge or color metric
        passed = passed_texture and passed_edges

        # Combined score (0-1) — weighted average of normalised metrics
        tex_score = min(1.0, texture_variance / (TEXTURE_VARIANCE_MIN * 2))
        edge_score = min(1.0, edge_strength / (EDGE_STRENGTH_MIN * 2))
        col_score = min(1.0, color_spread / 40.0)
        combined = (tex_score * 0.5) + (edge_score * 0.3) + (col_score * 0.2)

        details = (
            f"tex_var={texture_variance:.1f} "
            f"(thresh>={TEXTURE_VARIANCE_MIN}), "
            f"edge={edge_strength:.1f} "
            f"(thresh>={EDGE_STRENGTH_MIN}), "
            f"color_sprd={color_spread:.1f}"
        )

        if _DEBUG:
            print(f"[TextureLiveness] {details}")

        print(f"[Liveness] Texture variance: {texture_variance:.1f} "
              f"{'✅' if passed_texture else '❌'} "
              f"(threshold >= {TEXTURE_VARIANCE_MIN})")
        print(f"[Liveness] Edge strength: {edge_strength:.1f} "
              f"{'✅' if passed_edges else '❌'} "
              f"(threshold >= {EDGE_STRENGTH_MIN})")
        print(f"[Liveness] Color spread: {color_spread:.1f}")
        outcome = "PASSED ✅" if passed else "FAILED ❌"
        print(f"[Liveness] {outcome} (score={combined:.3f})")

        return {
            'passed': passed,
            'score': round(combined, 3),
            'texture_variance': round(texture_variance, 1),
            'edge_strength': round(edge_strength, 1),
            'color_spread': round(color_spread, 1),
            'details': details,
        }
