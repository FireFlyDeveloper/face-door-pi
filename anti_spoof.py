"""
anti_spoof.py — Single-frame liveness detection for face anti-spoofing.

Two backends:
  1. MobileNetV2 via TFLite — primary, catches printed photo + phone screen
  2. LBP texture fallback   — classical method when no TFLite model available

Reference:
  ScienceDirect (2024): MobileNetV2 transfer learning achieves 96% accuracy
  on live subjects, <0.6s total pipeline on Raspberry Pi 4B.

Model source:
  MiniFASNet: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
  Convert to TFLite via scripts/convert_model.py or download from:
  https://github.com/nicknochnius/Face-Antispoofing/releases
"""

from __future__ import annotations

import cv2
import numpy as np
import os
import logging
from typing import Optional

log = logging.getLogger(__name__)

TFLITE_MODEL = "./models/anti_spoof_mobilenetv2.tflite"
INPUT_SIZE = (224, 224)

# ImageNet normalization for MobileNetV2
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Threshold: score >= LIVE_THRESHOLD → LIVE
LIVE_THRESHOLD = 0.5


class AntiSpoofDetector:
    """
    Returns liveness probability [0.0 - 1.0].
    Score >= LIVE_THRESHOLD → LIVE
    Score <  LIVE_THRESHOLD → SPOOF
    """

    def __init__(self):
        self.interpreter: Optional[object] = None
        self.input_idx: Optional[int] = None
        self.output_idx: Optional[int] = None
        self._load_model()

    def _load_model(self):
        if not os.path.exists(TFLITE_MODEL):
            log.warning(
                "AntiSpoofDetector: TFLite model not found at %s — "
                "using LBP fallback. Run scripts/download_models.sh",
                TFLITE_MODEL,
            )
            return

        try:
            import tflite_runtime.interpreter as tflite
            self.interpreter = tflite.Interpreter(model_path=TFLITE_MODEL)
        except ImportError:
            try:
                import tensorflow as tf
                self.interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
            except ImportError:
                log.error(
                    "Neither tflite_runtime nor tensorflow found. "
                    "Run: pip install tflite-runtime"
                )
                return

        self.interpreter.allocate_tensors()
        inp = self.interpreter.get_input_details()
        self.input_idx = inp[0]["index"]
        self.output_idx = self.interpreter.get_output_details()[0]["index"]

        # Log input shape for debugging
        input_shape = inp[0]["shape"]
        log.info(
            "AntiSpoofDetector: TFLite MobileNetV2 loaded (input=%s)",
            input_shape,
        )

    def predict(self, face_img: np.ndarray) -> float:
        """
        Returns liveness score: 1.0 = definitely live, 0.0 = definitely spoof.
        """
        if self.interpreter is not None and self.input_idx is not None:
            return self._predict_tflite(face_img)
        return self._predict_lbp(face_img)

    def _predict_tflite(self, face_img: np.ndarray) -> float:
        # Preprocess
        img = cv2.resize(face_img, INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

        self.interpreter.set_tensor(self.input_idx, img)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_idx)

        # Output: [spoof_prob, live_prob] or single logit
        if output.shape[-1] == 2:
            live_score = float(output[0][1])  # index 1 = live class
        else:
            live_score = float(output[0][0])  # sigmoid output

        score = np.clip(live_score, 0.0, 1.0)
        log.debug("AntiSpoof TFLite: score=%.3f", score)
        return float(score)

    def _predict_lbp(self, face_img: np.ndarray) -> float:
        """
        Classical LBP texture fallback.

        Real faces have higher texture variance than printed/screen images.
        Reference: Boulkenafet et al. (2016) — Color LBP anti-spoofing.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        lbp = self._compute_lbp(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-6

        # Entropy as proxy for texture richness
        entropy = -float(np.sum(hist * np.log2(hist + 1e-10)))

        # Empirical: real faces ~6.5-7.5, print/screen lower
        LIVE_ENTROPY_THRESHOLD = 6.2
        score = np.clip((entropy - LIVE_ENTROPY_THRESHOLD) / 2.0, 0.0, 1.0)
        log.debug("AntiSpoof LBP: entropy=%.2f score=%.3f", entropy, score)
        return float(score)

    @staticmethod
    def _compute_lbp(gray: np.ndarray) -> np.ndarray:
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
