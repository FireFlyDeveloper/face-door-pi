"""
anti_spoof.py — Single-frame liveness detection for face anti-spoofing.

Three backends (tried in order):
  1. MiniFASNet ONNX  — primary (via onnxruntime, already on Pi)
  2. MobileNetV2 TFLite — secondary
  3. LBP texture       — classical fallback

Reference:
  ScienceDirect (2024): MobileNetV2 transfer learning achieves 96% accuracy
  on live subjects, <0.6s total pipeline on Raspberry Pi 4B.

Models:
  MiniFASNet: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing
"""

from __future__ import annotations

import cv2
import numpy as np
import os
import logging
from typing import Optional

log = logging.getLogger(__name__)

ONNX_MODEL = "./models/minifasnet_v2.onnx"
TFLITE_MODEL = "./models/anti_spoof_mobilenetv2.tflite"
ONNX_INPUT_SIZE = (80, 80)     # MiniFASNet input
TFLITE_INPUT_SIZE = (224, 224)  # MobileNetV2 input

# Threshold: score >= LIVE_THRESHOLD → LIVE
LIVE_THRESHOLD = 0.5


class AntiSpoofDetector:
    """
    Returns liveness probability [0.0 - 1.0].
    Score >= LIVE_THRESHOLD → LIVE
    Score <  LIVE_THRESHOLD → SPOOF
    """

    def __init__(self):
        self._backend = "lbp"  # fallback default
        self._session: Optional[object] = None
        self._input_name: Optional[str] = None
        self._input_size = None
        self._tflite_interpreter: Optional[object] = None
        self._tflite_input_idx: Optional[int] = None
        self._tflite_output_idx: Optional[int] = None
        self._load_model()

    def _load_model(self):
        """Try ONNX first, then TFLite, fall back to LBP."""
        # Try ONNX (MiniFASNet)
        if os.path.exists(ONNX_MODEL):
            try:
                import onnxruntime as ort
                opts = ort.SessionOptions()
                opts.inter_op_num_threads = 4
                opts.intra_op_num_threads = 4
                self._session = ort.InferenceSession(
                    ONNX_MODEL,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._session.get_inputs()[0].name
                self._input_size = ONNX_INPUT_SIZE
                self._backend = "onnx"
                log.info(
                    "AntiSpoofDetector: ONNX MiniFASNet loaded (%s)",
                    ONNX_MODEL,
                )
                return
            except Exception as e:
                log.warning("AntiSpoofDetector: ONNX load failed: %s", e)

        # Try TFLite (MobileNetV2)
        if os.path.exists(TFLITE_MODEL):
            try:
                import tflite_runtime.interpreter as tflite
                self._tflite_interpreter = tflite.Interpreter(
                    model_path=TFLITE_MODEL
                )
            except ImportError:
                try:
                    import tensorflow as tf
                    self._tflite_interpreter = tf.lite.Interpreter(
                        model_path=TFLITE_MODEL
                    )
                except ImportError:
                    pass

            if self._tflite_interpreter is not None:
                self._tflite_interpreter.allocate_tensors()
                inp = self._tflite_interpreter.get_input_details()
                self._tflite_input_idx = inp[0]["index"]
                self._tflite_output_idx = \
                    self._tflite_interpreter.get_output_details()[0]["index"]
                self._backend = "tflite"
                log.info(
                    "AntiSpoofDetector: TFLite MobileNetV2 loaded (%s)",
                    TFLITE_MODEL,
                )
                return

        # LBP fallback
        log.warning(
            "AntiSpoofDetector: No model found — using LBP fallback. "
            "Run scripts/download_models.sh or place model in models/"
        )

    def predict(self, face_img: np.ndarray) -> float:
        """
        Returns liveness score: 1.0 = definitely live, 0.0 = definitely spoof.
        """
        if self._backend == "onnx":
            return self._predict_onnx(face_img)
        elif self._backend == "tflite":
            return self._predict_tflite(face_img)
        return self._predict_lbp(face_img)

    def _predict_onnx(self, face_img: np.ndarray) -> float:
        """MiniFASNet ONNX inference."""
        inputs = self._preprocess_onnx(face_img)
        output = self._session.run(None, {self._input_name: inputs})

        # MiniFASNet outputs: [spoof, ?, live] or [spoof_spoof, ~, live]
        out = output[0]
        if out.shape[-1] >= 3:
            # 3-class: index 2 discriminates but in REVERSE
            # Photo (fake) → index 2 ≈ 0.9, Real face → index 2 ≈ 0.1
            # So live_score = 1.0 - softmax_index_2
            scores = out[0]
            exp_scores = np.exp(scores - np.max(scores))
            idx2_score = float(exp_scores[2] / np.sum(exp_scores))
            live_score = 1.0 - idx2_score
        elif out.shape[-1] == 2:
            scores = out[0]
            exp_scores = np.exp(scores - np.max(scores))
            live_score = float(exp_scores[1] / np.sum(exp_scores))
        else:
            live_score = float(1.0 / (1.0 + np.exp(-out[0][0])))  # sigmoid

        score = np.clip(live_score, 0.0, 1.0)
        log.debug("AntiSpoof ONNX: score=%.3f", score)
        return score

    def _preprocess_onnx(self, face_img: np.ndarray) -> np.ndarray:
        """Preprocess face crop for MiniFASNet ONNX.

        NOTE: MiniFASNet was trained with [0, 255] float inputs (no /255.0).
        The original ``to_tensor()`` in Silent-Face-Anti-Spoofing returns raw
        uint8 values cast to float32 without dividing by 255.
        """
        w, h = ONNX_INPUT_SIZE
        img = cv2.resize(face_img, (w, h))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        # NO /255.0 — MiniFASNet expects [0, 255] range
        # MiniFASNet uses (1, 3, H, W) format (NCHW)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        return img

    def _predict_tflite(self, face_img: np.ndarray) -> float:
        """MobileNetV2 TFLite inference."""
        img = cv2.resize(face_img, TFLITE_INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        # ImageNet norm
        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - MEAN) / STD
        img = np.expand_dims(img, axis=0)

        self._tflite_interpreter.set_tensor(
            self._tflite_input_idx, img
        )
        self._tflite_interpreter.invoke()
        output = self._tflite_interpreter.get_tensor(
            self._tflite_output_idx
        )

        if output.shape[-1] == 2:
            live_score = float(output[0][1])
        else:
            live_score = float(output[0][0])

        score = np.clip(live_score, 0.0, 1.0)
        log.debug("AntiSpoof TFLite: score=%.3f", score)
        return score

    def _predict_lbp(self, face_img: np.ndarray) -> float:
        """
        Classical LBP texture fallback.

        Reference: Boulkenafet et al. (2016) — Color LBP anti-spoofing.
        """
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (64, 64))

        lbp = self._compute_lbp(gray)
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        hist /= hist.sum() + 1e-6

        entropy = -float(np.sum(hist * np.log2(hist + 1e-10)))
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
