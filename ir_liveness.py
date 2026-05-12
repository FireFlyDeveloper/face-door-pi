"""
ir_liveness.py — Multi-frame encoding consistency anti-spoof liveness.

Principle:
  A live person has natural micro-movements (breathing, posture sway)
  between consecutive frames, causing the 128-D face encoding to vary
  slightly. A held phone screen or printed photo is unnaturally static,
  producing near-identical encodings across frames.

  This is a well-documented temporal anti-spoofing technique known as
  "frame-to-frame encoding consistency" in liveness detection literature.

Metrics (across N consecutive frames):
  1. Mean pairwise encoding distance — live > threshold
  2. Encoding vector variance (std per dimension, then mean)

Public interface:
  IRLivenessDetector.check_liveness(frames, face_rect, face_recognizer) -> dict
  IRLivenessDetector.reset()
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional


# ── Thresholds ─────────────────────────────────────────────────────────
# Minimum mean pairwise distance between face encodings across frames
# Live faces exhibit natural micro-movement variation > 0.04
# Static screens produce near-identical encodings < 0.02
ENCODING_VARIANCE_MIN = 0.04
# Minimum number of frames required
MIN_FRAMES = 2


_DEBUG = False


class IRLivenessDetector:
    """Multi-frame encoding consistency anti-spoof liveness detector."""

    def __init__(self, debug: bool = False):
        global _DEBUG
        _DEBUG = debug
        self.reset()

    def reset(self):
        """Clear state between checks."""
        self._encodings: List[np.ndarray] = []
        self._frame_count = 0

    def check_liveness(
        self,
        frames: List[np.ndarray],
        face_rect,
        face_recognizer,
    ) -> Dict:
        """Analyze encoding consistency across multiple frames.

        Args:
            frames: List of BGR numpy arrays (consecutive captures).
            face_rect: dlib rectangle (same rect used for all frames).
            face_recognizer: FaceRecognizer instance for encoding.

        Returns:
            dict with: passed, score, metrics, details.
        """
        if len(frames) < MIN_FRAMES:
            return {
                'passed': False,
                'score': 0.0,
                'mean_encoding_dist': 0.0,
                'encoding_std': 0.0,
                'details': f'insufficient frames ({len(frames)} < {MIN_FRAMES})',
            }

        self.reset()

        # Extract encoding from each frame using the face recognizer
        for frame in frames:
            result = face_recognizer.get_face_encoding(frame)
            if result is not None:
                encoding, _ = result
                self._encodings.append(encoding)

        if len(self._encodings) < MIN_FRAMES:
            return {
                'passed': False,
                'score': 0.0,
                'mean_encoding_dist': 0.0,
                'encoding_std': 0.0,
                'details': f'face encoding failed in {len(self._encodings)}/{len(frames)} frames',
            }

        # ── Metric 1: Mean pairwise encoding distance ────────────
        pairwise_dists = []
        for i in range(len(self._encodings)):
            for j in range(i + 1, len(self._encodings)):
                dist = float(np.linalg.norm(self._encodings[i] - self._encodings[j]))
                pairwise_dists.append(dist)
        mean_encoding_dist = float(np.mean(pairwise_dists)) if pairwise_dists else 0.0

        # ── Metric 2: Encoding vector variance ───────────────────
        stacked = np.stack(self._encodings, axis=0)
        per_dim_std = np.std(stacked, axis=0)
        encoding_std = float(np.mean(per_dim_std))

        # ── Decision ─────────────────────────────────────────────
        passed = mean_encoding_dist >= ENCODING_VARIANCE_MIN

        # Normalised score
        score = min(1.0, mean_encoding_dist / (ENCODING_VARIANCE_MIN * 3))

        print(f"[EncodingLiveness] {len(self._encodings)} encodings across "
              f"{len(frames)} frames")
        print(f"[EncodingLiveness] Mean pairwise distance: {mean_encoding_dist:.4f} "
              f"{'✅' if passed else '❌'} (>= {ENCODING_VARIANCE_MIN})")
        print(f"[EncodingLiveness] Encoding std: {encoding_std:.4f}")
        print(f"[EncodingLiveness] Pairwise distances: "
              f"{[f'{d:.4f}' for d in pairwise_dists]}")
        outcome = "PASSED ✅" if passed else "FAILED ❌"
        print(f"[EncodingLiveness] {outcome} (score={score:.3f})")

        return {
            'passed': passed,
            'score': round(score, 3),
            'mean_encoding_dist': round(mean_encoding_dist, 4),
            'encoding_std': round(encoding_std, 4),
            'details': (
                f"encodings={len(self._encodings)}, "
                f"mean_dist={mean_encoding_dist:.4f} "
                f"(thresh>={ENCODING_VARIANCE_MIN})"
            ),
        }
