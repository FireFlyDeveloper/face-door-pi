"""
anti_spoof.py — Single-frame liveness detection for face anti-spoofing.

Two backends fused by weighted average:
  1. MiniFASNet ONNX (weight 0.7)  — primary deep learning signal
  2. LBP texture      (weight 0.3)  — classical texture backup

MiniFASNet on NoIR v2: index-2 softmax is HIGH for spoof (≈0.9), LOW for live (≈0.1).
  → live_score = 1.0 - softmax_index_2

LBP on NoIR v2: photos have HIGHER entropy than live faces (reverse of standard cameras).
  → live_score = clamp((threshold - entropy) / margin)

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
log.setLevel(logging.DEBUG)

ONNX_MODEL = "./models/minifasnet_v2.onnx"
ONNX_INPUT_SIZE = (80, 80)

# Weights for the ensemble
ONNX_WEIGHT = 0.7
LBP_WEIGHT = 0.3

# LBP thresholds (calibrated for NoIR v2)
LBP_THRESHOLD = 6.35     # midpoint estimate
LBP_MARGIN = 0.35

# Final threshold
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
        if not os.path.exists(ONNX_MODEL):
            log.info("No ONNX model at %s — LBP only", ONNX_MODEL)
            return
        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 4
            opts.intra_op_num_threads = 4
            self._onnx_session = ort.InferenceSession(
                ONNX_MODEL, sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self._onnx_input_name = self._onnx_session.get_inputs()[0].name
            self._has_onnx = True
            log.info("MiniFASNet ONNX loaded (%s)", ONNX_MODEL)
        except Exception as e:
            log.warning("ONNX load failed: %s", e)

    def predict(self, face_img: np.ndarray) -> float:
        """Fused liveness score: ONNX (%.0f%%) + LBP (%.0f%%)."""
        onnx_score = self._predict_onnx(face_img) if self._has_onnx else 0.0
        lbp_score = self._predict_lbp(face_img)

        if self._has_onnx:
            combined = ONNX_WEIGHT * onnx_score + LBP_WEIGHT * lbp_score
            log.debug("AntiSpoof: ONNX=%.3f LBP=%.3f combined=%.3f",
                      onnx_score, lbp_score, combined)
            return float(np.clip(combined, 0.0, 1.0))

        log.debug("AntiSpoof LBP: score=%.3f", lbp_score)
        return float(np.clip(lbp_score, 0.0, 1.0))

    # ── LBP ──────────────────────────────────────

    def _predict_lbp(self, face_img: np.ndarray) -> float:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        lbp = self._compute_lbp(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)

        entropy = -float(np.sum(hist * np.log2(hist + 1e-10)))
        score = np.clip((LBP_THRESHOLD - entropy) / LBP_MARGIN, 0.0, 1.0)
        return float(score)

    @staticmethod
    def _compute_lbp(gray):
        lbp = np.zeros_like(gray, dtype=np.uint8)
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
                     (1, 1), (1, 0), (1, -1), (0, -1)]
        center = gray[1:-1, 1:-1]
        for i, (dy, dx) in enumerate(neighbors):
            ny, nx = 1 + dy, 1 + dx
            n = gray[ny:ny + gray.shape[0] - 2, nx:nx + gray.shape[1] - 2]
            lbp[1:-1, 1:-1] |= ((n >= center).astype(np.uint8) << i)
        return lbp

    # ── ONNX ─────────────────────────────────────

    def _predict_onnx(self, face_img: np.ndarray) -> float:
        inputs = self._preprocess_onnx(face_img)
        output = self._onnx_session.run(
            None, {self._onnx_input_name: inputs}
        )
        out = output[0]
        if out.shape[-1] >= 3:
            scores = out[0]
            e = np.exp(scores - np.max(scores))
            idx2 = float(e[2] / np.sum(e))
            live_score = 1.0 - idx2
        elif out.shape[-1] == 2:
            scores = out[0]
            e = np.exp(scores - np.max(scores))
            live_score = float(e[1] / np.sum(e))
        else:
            live_score = float(1.0 / (1.0 + np.exp(-out[0][0])))

        return float(np.clip(live_score, 0.0, 1.0))

    def _preprocess_onnx(self, face_img):
        w, h = ONNX_INPUT_SIZE
        img = cv2.resize(face_img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)  # [0, 255], NO /255.0
        img = np.transpose(img, (2, 0, 1))[None, :]
        return img
