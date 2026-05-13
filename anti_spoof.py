"""
anti_spoof.py — Single-frame liveness detection for face anti-spoofing.

Ensemble: MiniFASNet ONNX (weight 0.85) + Gray LBP entropy (weight 0.15).
LBP is unreliable on NoIR camera (IR flood flips entropy direction), so
ONNX is the primary signal. LBP acts as a minor sanity check only.

Calibrated on NoIR v2 camera (640x480), 2026-05-13:
  Metric            Photo     Live     Gap
  ONNX idx2         0.8485    0.3522   0.4963  ← reliable
  ONNX inverted     0.1515    0.6478   0.4963
  LBP gray entropy  6.2911    5.3504   -0.9407 ← flips with lighting

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

# ── Ensemble weights ────────────────────────────
# ONNX is the primary signal on NoIR camera (LBP flips with IR)
ONNX_WEIGHT = 0.85
LBP_WEIGHT = 0.15

# ── LBP thresholds ──────────────────────────────
# NoIR: LBP entropy is unreliable (direction flips with IR lighting)
# Used as minor sanity check only (15% weight)
LBP_THRESHOLD = 5.82     # midpoint from latest calibration
LBP_MARGIN = 0.60        # maps (entropy - threshold) / margin to [0,1]

# ── Final threshold ─────────────────────────────
# Calibrated 2026-05-13: photo=0.14, live=0.43 @ ONNX 85%/LBP 15%
# Midpoint between photo and live ensemble scores ≈ 0.25
SCORE_THRESHOLD = 0.25


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
        """Ensemble: ONNX inverted + LBP entropy."""
        onnx_score = self._predict_onnx(face_img) if self._has_onnx else 0.0
        lbp_score = self._predict_lbp(face_img)

        if self._has_onnx:
            combined = ONNX_WEIGHT * onnx_score + LBP_WEIGHT * lbp_score
            log.info(
                "AntiSpoof: ONNX=%.3f LBP=%.3f combined=%.3f",
                onnx_score, lbp_score, combined,
            )
            return float(np.clip(combined, 0.0, 1.0))

        log.info("AntiSpoof LBP: score=%.3f", lbp_score)
        return float(np.clip(lbp_score, 0.0, 1.0))

    # ── LBP (primary texture) ──────────────────

    def _predict_lbp(self, face_img: np.ndarray) -> float:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        lbp = self._compute_lbp(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32) / (hist.sum() + 1e-6)

        entropy = -float(np.sum(hist * np.log2(hist + 1e-10)))
        # Higher entropy = LIVE (classical LBP — rich texture)
        score = np.clip((entropy - LBP_THRESHOLD) / LBP_MARGIN, 0.0, 1.0)
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

    # ── MiniFASNet ONNX (deep secondary signal) ─

    def _predict_onnx(self, face_img: np.ndarray) -> float:
        inputs = self._preprocess_onnx(face_img)
        output = self._onnx_session.run(
            None, {self._onnx_input_name: inputs}
        )
        out = output[0]
        if out.shape[-1] >= 3:
            scores = out[0]
            e = np.exp(scores - np.max(scores))
            # On NoIR: idx2=0.84(spoof) / 0.41(live) → invert
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
        img = img.astype(np.float32)  # [0,255] — NO /255.0
        img = np.transpose(img, (2, 0, 1))[None, :]
        return img
