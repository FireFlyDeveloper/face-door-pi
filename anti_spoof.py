"""
anti_spoof.py — Single-frame liveness detection for face anti-spoofing.

Two backends used together:
  1. LBP texture analysis     — primary (works on any camera, NoIR-compatible)
  2. MiniFASNet ONNX          — secondary (deep learning signal)

Combined score = average of both signals for robustness.

NoIR v2 camera behaviour (empirically determined):
  PHOTO: higher LBP entropy (paper texture visible in IR) → SPOOF
  LIVE:  lower LBP entropy (skin looks smoother in IR)    → LIVE
  This is the REVERSE of standard RGB cameras.

Reference:
  Boulkenafet et al. (2016) — Color LBP anti-spoofing.
  Minivision (2019) — Silent-Face-Anti-Spoofing (MiniFASNet).
"""

from __future__ import annotations

import cv2
import numpy as np
import os
import logging
from typing import Optional

log = logging.getLogger(__name__)

ONNX_MODEL = "./models/minifasnet_v2.onnx"
ONNX_INPUT_SIZE = (80, 80)

# LBP entropy thresholds (calibrated for NoIR v2 at 640x480)
# Lower entropy = LIVE, Higher entropy = SPOOF
LBP_THRESHOLD = 6.37    # midpoint between photo(6.51) and live(6.23)
LBP_MARGIN = 0.30       # map (threshold - entropy) / margin to [0, 1]

# Final threshold: combined score >= SCORE_THRESHOLD → LIVE
SCORE_THRESHOLD = 0.5


class AntiSpoofDetector:
    """
    Returns liveness probability [0.0 - 1.0].
    Score >= SCORE_THRESHOLD → LIVE
    Score <  SCORE_THRESHOLD → SPOOF
    """

    def __init__(self):
        self._onnx_session: Optional[object] = None
        self._onnx_input_name: Optional[str] = None
        self._has_onnx = False
        self._load_onnx()

    def _load_onnx(self):
        """Load MiniFASNet ONNX model if available (secondary signal)."""
        if not os.path.exists(ONNX_MODEL):
            log.info("AntiSpoofDetector: No ONNX model at %s — using LBP only", ONNX_MODEL)
            return
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 4
            opts.intra_op_num_threads = 4
            self._onnx_session = ort.InferenceSession(
                ONNX_MODEL,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._onnx_input_name = self._onnx_session.get_inputs()[0].name
            self._has_onnx = True
            log.info(
                "AntiSpoofDetector: MiniFASNet ONNX loaded (%s) as secondary signal",
                ONNX_MODEL,
            )
        except Exception as e:
            log.warning("AntiSpoofDetector: ONNX load failed: %s", e)

    def predict(self, face_img: np.ndarray) -> float:
        """
        Returns liveness score [0.0, 1.0].
        Uses LBP (always available) + MiniFASNet ONNX (if loaded).
        """
        lbp_score = self._predict_lbp(face_img)

        if self._has_onnx:
            onnx_score = self._predict_onnx(face_img)
            combined = (lbp_score + onnx_score) / 2.0
            log.debug(
                "AntiSpoof: LBP=%.3f ONNX=%.3f combined=%.3f",
                lbp_score, onnx_score, combined,
            )
            return float(np.clip(combined, 0.0, 1.0))

        log.debug("AntiSpoof LBP: score=%.3f", lbp_score)
        return float(np.clip(lbp_score, 0.0, 1.0))

    # ──────────────────────────────────────────────
    # LBP texture analysis (primary)
    # ──────────────────────────────────────────────

    def _predict_lbp(self, face_img: np.ndarray) -> float:
        """
        LBP texture analysis.

        On NoIR v2: photos have HIGHER entropy than live faces.
        So score = clamp((LBP_THRESHOLD - entropy) / LBP_MARGIN, 0, 1).
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        lbp = self._compute_lbp(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-6

        entropy = -float(np.sum(hist * np.log2(hist + 1e-10)))
        score = np.clip(
            (LBP_THRESHOLD - entropy) / LBP_MARGIN, 0.0, 1.0
        )
        log.debug("AntiSpoof LBP: entropy=%.4f score=%.3f", entropy, score)
        return float(score)

    @staticmethod
    def _compute_lbp(gray: np.ndarray) -> np.ndarray:
        """Basic 8-neighbour LBP."""
        lbp = np.zeros_like(gray, dtype=np.uint8)
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                     (1, 1), (1, 0), (1, -1), (0, -1)]
        center = gray[1:-1, 1:-1]
        for i, (dy, dx) in enumerate(neighbors):
            ny, nx = 1 + dy, 1 + dx
            neighbor = gray[ny:ny + gray.shape[0] - 2,
                            nx:nx + gray.shape[1] - 2]
            lbp[1:-1, 1:-1] |= ((neighbor >= center).astype(np.uint8) << i)
        return lbp

    # ──────────────────────────────────────────────
    # MiniFASNet ONNX (secondary, deep learning)
    # ──────────────────────────────────────────────

    def _predict_onnx(self, face_img: np.ndarray) -> float:
        """MiniFASNet ONNX inference.

        On NoIR: index 2 is HIGH for spoof and LOW for live.
        So live_score = 1.0 - softmax_index_2.
        """
        inputs = self._preprocess_onnx(face_img)
        output = self._onnx_session.run(
            None, {self._onnx_input_name: inputs}
        )

        out = output[0]
        if out.shape[-1] >= 3:
            scores = out[0]
            exp_scores = np.exp(scores - np.max(scores))
            idx2 = float(exp_scores[2] / np.sum(exp_scores))
            live_score = 1.0 - idx2
        elif out.shape[-1] == 2:
            scores = out[0]
            exp_scores = np.exp(scores - np.max(scores))
            live_score = float(exp_scores[1] / np.sum(exp_scores))
        else:
            live_score = float(1.0 / (1.0 + np.exp(-out[0][0])))

        score = np.clip(live_score, 0.0, 1.0)
        log.debug("AntiSpoof ONNX: score=%.3f", score)
        return score

    def _preprocess_onnx(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face crop for MiniFASNet ONNX ([0,255] float, NCHW)."""
        w, h = ONNX_INPUT_SIZE
        img = cv2.resize(face_img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        # NO /255.0 — MiniFASNet expects [0, 255] range
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img
