"""
ir_liveness.py — NoIR spectral liveness detection (no external IR LED needed).

Uses the Pi NoIR camera's inherent IR sensitivity (no IR-cut filter) to
distinguish live faces from photos/screens by analyzing the spectral
signature of the face region in a single frame.

Principle:
  The NoIR sensor captures NIR (near-infrared) light that bleeds into all
  RGB channels, but MOST strongly into the RED channel (standard Bayer
  pattern behavior). Real skin reflects NIR strongly, so the R channel is
  boosted compared to G/B. Printed photos and phone screens do not reflect
  NIR the same way, producing a different spectral balance.

Metrics (computed on face region):
  1. Red Dominance Ratio: R / (G + B + eps)  — higher for live skin (NIR boost)
  2. R-G Mean Difference: mean(R) - mean(G)  — positive is typical for skin
  3. Red Excess Index: (2*R - G - B) / (R + G + B + eps)  — normalised

Public interface:
  check_liveness(frame, face_rect) -> dict
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Optional

# ── Thresholds (tune empirically) ───────────────────────────────────────
# Under NoIR, live skin typically has R channel boosted by NIR bleed
RED_DOMINANCE_MIN = 0.38      # min R/(G+B) ratio for live skin
RED_EXCESS_MIN = 0.0          # min (2R-G-B)/(R+G+B) normalised excess
MIN_FACE_REGION = 100         # minimum face crop pixels

# Debug: print per-frame metrics
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
    """NoIR spectral liveness detector — single-frame, no external IR LED."""

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
        """Analyze a single BGR frame for NoIR spectral liveness.

        Args:
            frame: BGR numpy array (from Pi NoIR camera).
            face_rect: dlib rectangle of the detected face.

        Returns:
            dict with: passed, score, metrics, details.
        """
        crop = _face_region(frame, face_rect)
        if crop is None:
            return {
                'passed': False,
                'score': 0.0,
                'red_dominance': 0.0,
                'red_excess': 0.0,
                'details': 'face region too small',
            }

        # Split channels and convert to float
        b = crop[:, :, 0].astype(np.float32)
        g = crop[:, :, 1].astype(np.float32)
        r = crop[:, :, 2].astype(np.float32)

        mean_r = float(np.mean(r))
        mean_g = float(np.mean(g))
        mean_b = float(np.mean(b))
        total = mean_r + mean_g + mean_b + 1e-8

        # ── Metric 1: Red Dominance ────────────────────────────────
        # R/(G+B) — live skin gets NIR boost in R channel
        red_dominance = mean_r / (mean_g + mean_b + 1e-8)

        # ── Metric 2: R-G mean difference ──────────────────────────
        rg_diff = mean_r - mean_g

        # ── Metric 3: Normalised Red Excess ─────────────────────────
        # (2R - G - B) / (R + G + B) — standard vegetation index adapted
        red_excess = (2.0 * mean_r - mean_g - mean_b) / total

        # ── Decision ───────────────────────────────────────────────
        passed_dominance = red_dominance >= RED_DOMINANCE_MIN
        passed_excess = red_excess >= RED_EXCESS_MIN

        # Combined score: weighted average of normalised metrics
        dom_score = max(0.0, min(1.0, red_dominance / (RED_DOMINANCE_MIN * 1.5)))
        # Avoid division by zero when RED_EXCESS_MIN == 0
        if RED_EXCESS_MIN > 0:
            exc_score = max(0.0, min(1.0, red_excess / (RED_EXCESS_MIN * 1.5)))
        else:
            exc_score = 1.0 if red_excess >= 0 else 0.0
        combined = (dom_score * 0.6) + (exc_score * 0.4)

        passed = passed_dominance and passed_excess

        details = (
            f"R/G/B={mean_r:.0f}/{mean_g:.0f}/{mean_b:.0f} "
            f"R/(G+B)={red_dominance:.3f} "
            f"(thresh>={RED_DOMINANCE_MIN}), "
            f"R-G={rg_diff:.1f}, "
            f"red_excess={red_excess:.3f} "
            f"(thresh>={RED_EXCESS_MIN})"
        )

        if _DEBUG:
            print(f"[IR] {details}")

        return {
            'passed': passed,
            'score': combined,
            'red_dominance': red_dominance,
            'rg_diff': rg_diff,
            'red_excess': red_excess,
            'details': details,
        }
