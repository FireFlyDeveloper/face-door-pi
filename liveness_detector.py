"""
liveness_detector.py — MiniFASNet ONNX anti-spoofing.

Replaces EAR blink detector with a proper face anti-spoofing model:
  MiniFASNet-V2 ONNX (80x80 BGR input, 3-class softmax)
  [live, print-attack, replay-attack]

Single-frame inference. No PyTorch needed — runs on ONNX Runtime.

Public interface:
  reset()              — clear state
  process_frame(frame) — per-frame inference → dict
  check_liveness(frames) — batch mode → dict
"""

import os
from typing import Dict, List, Optional

import cv2
import dlib
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────
SCORE_THRESHOLD = 0.5        # min liveness score to pass (0-1)
SCALE = 2.7                  # crop margin multiplier around bbox
INPUT_SIZE = 80              # model input (80x80)
SCORE_SMOOTH_ALPHA = 0.5     # EMA for per-frame scoring
MIN_FRAMES = 5               # min frames before passed=true

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "models",
    "minifasnet_v2.onnx",
)

SHAPE_PREDICTOR_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "shape_predictor_68_face_landmarks.dat",
)


def _crop_face(img: np.ndarray, face_rect, scale: float = SCALE) -> Optional[np.ndarray]:
    """Crop face region with margin around dlib rectangle."""
    h, w = img.shape[:2]
    cx = (face_rect.left() + face_rect.right()) / 2.0
    cy = (face_rect.top() + face_rect.bottom()) / 2.0
    face_w = face_rect.width()
    face_h = face_rect.height()
    size = max(face_w, face_h) * scale / 2.0

    x1 = int(max(cx - size, 0))
    y1 = int(max(cy - size, 0))
    x2 = int(min(cx + size, w))
    y2 = int(min(cy + size, h))

    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


class LivenessDetector:
    """MiniFASNet ONNX anti-spoof detector."""

    def __init__(self):
        self._session: Optional = None
        self._detector: Optional[dlib.frone_face_detector] = None
        self._input_name: Optional[str] = None
        self.reset()

    def _ensure_models(self):
        if self._session is not None:
            return

        # Load ONNX model
        import onnxruntime as ort
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"MiniFASNet model not found at: {MODEL_PATH}\n"
                f"Download from: https://huggingface.co/garciafido/minifasnet-v2-anti-spoofing-onnx"
            )
        self._session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

        # Load dlib face detector (lightweight, no shape predictor needed)
        self._detector = dlib.get_frontal_face_detector()

    def reset(self):
        """Reset scoring state."""
        self._smooth_score = 0.5
        self._frame_count = 0

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Process a single frame for anti-spoof.

        Args:
            frame: BGR numpy array.

        Returns:
            dict with: passed, score, liveness_score, face_detected, details.
        """
        self._ensure_models()
        self._frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._detector(gray, 0)

        if not faces:
            self._smooth_score = 0.5
            return {
                'passed': False,
                'score': 0.5,
                'liveness_score': 0.5,
                'face_detected': False,
                'details': 'no face',
            }

        face = max(faces, key=lambda r: r.width() * r.height())
        crop = _crop_face(frame, face)
        if crop is None or crop.size < 100:
            return {
                'passed': False,
                'score': 0.5,
                'liveness_score': 0.5,
                'face_detected': True,
                'details': 'crop too small',
            }

        # Preprocess: resize → normalize → NCHW
        resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE))
        normalized = resized.astype(np.float32) / 255.0
        nchw = np.transpose(normalized, (2, 0, 1))  # HWC → CHW
        batch = np.expand_dims(nchw, axis=0)         # → NCHW

        # Inference
        outputs = self._session.run(None, {self._input_name: batch})
        logits = outputs[0][0]  # 3-class: [live, print, replay]

        # Softmax
        exp = np.exp(logits - np.max(logits))
        probs = exp / np.sum(exp)

        # Liveness score = live class probability
        live_prob = float(probs[0])

        # Smooth score
        self._smooth_score = (
            SCORE_SMOOTH_ALPHA * live_prob +
            (1 - SCORE_SMOOTH_ALPHA) * self._smooth_score
        )

        passed = (
            self._frame_count >= MIN_FRAMES and
            round(self._smooth_score, 3) >= SCORE_THRESHOLD
        )

        print_attack = float(probs[1])
        replay_attack = float(probs[2])

        return {
            'passed': passed,
            'score': self._smooth_score,
            'liveness_score': live_prob,
            'face_detected': True,
            'details': (
                f"live={live_prob:.3f} "
                f"print={print_attack:.3f} "
                f"replay={replay_attack:.3f} "
                f"smoothed={self._smooth_score:.3f}"
            ),
        }

    def check_liveness(self, frames: List[np.ndarray]) -> Dict:
        """Batch mode — processes all frames, returns average result."""
        self.reset()
        scores = []
        for frame in frames:
            result = self.process_frame(frame)
            scores.append(result['score'])

        avg_score = np.mean(scores) if scores else 0.0
        passed = len(scores) >= MIN_FRAMES and round(avg_score, 3) >= SCORE_THRESHOLD

        return {
            'passed': passed,
            'score': avg_score,
            'blinks_detected': 0,
            'blinks_required': 0,
            'details': f'avg_liveness={avg_score:.3f} over {len(scores)} frames',
        }
