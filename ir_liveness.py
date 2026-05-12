"""
ir_liveness.py — High-res texture/edge anti-fool liveness detection.

Uses the IMX219 at 1640x1232 (full sensor detail) to capture enough
texture information to distinguish live faces from printed photos or
phone screen replays via:

1. Texture variance (Laplacian): Real skin has fine pores, hair, and
   micro-texture at full resolution. Printed paper is flat/smooth.
   Phone screens have pixel grid artifacts at high magnification.
2. Edge sharpness (Sobel): Natural skin edges vs artificial reproductions.
3. Color saturation spread: Natural skin color distribution.

Public interface:
  IRLivenessDetector.check_liveness(frame, face_rect) -> dict
  IRLivenessDetector.reset()
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Optional


# ── Thresholds (tune empirically for Pi Cam v2 @ 1640x1232) ────────────
TEXTURE_VARIANCE_MIN = 3.0     # Laplacian variance
EDGE_STRENGTH_MIN = 5.0        # Mean Sobel magnitude
MIN_FACE_REGION = 100          # minimum face crop pixels


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
    """High-res texture/edge anti-fool liveness detector."""

    def __init__(self, debug: bool = False):
        global _DEBUG
        _DEBUG = debug
        self.reset()

    def reset(self):
        """Clear between checks."""
        pass

    def check_liveness(
        self,
        frame: np.ndarray,
        face_rect,
    ) -> Dict:
        """Analyze a single high-res BGR frame for texture-based liveness.

        Args:
            frame: BGR numpy array (1640x1232 recommended).
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
        passed = passed_texture and passed_edges

        tex_score = min(1.0, texture_variance / (TEXTURE_VARIANCE_MIN * 3))
        edge_score = min(1.0, edge_strength / (EDGE_STRENGTH_MIN * 3))
        col_score = min(1.0, color_spread / 40.0)
        combined = (tex_score * 0.5) + (edge_score * 0.3) + (col_score * 0.2)

        details = (
            f"tex_var={texture_variance:.1f} "
            f"(thresh>={TEXTURE_VARIANCE_MIN}), "
            f"edge={edge_strength:.1f} "
            f"(thresh>={EDGE_STRENGTH_MIN}), "
            f"color_sprd={color_spread:.1f}"
        )

        print(f"[TextureLiveness] tex_var={texture_variance:.1f} "
              f"{'✅' if passed_texture else '❌'} (>= {TEXTURE_VARIANCE_MIN})")
        print(f"[TextureLiveness] edge={edge_strength:.1f} "
              f"{'✅' if passed_edges else '❌'} (>= {EDGE_STRENGTH_MIN})")
        print(f"[TextureLiveness] color_spread={color_spread:.1f}")
        print(f"[TextureLiveness] {'PASSED ✅' if passed else 'FAILED ❌'} "
              f"(score={combined:.3f})")

        return {
            'passed': passed,
            'score': round(combined, 3),
            'texture_variance': round(texture_variance, 1),
            'edge_strength': round(edge_strength, 1),
            'color_spread': round(color_spread, 1),
            'details': details,
        }
