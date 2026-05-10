"""
liveness_detector.py — 3-layer anti-spoofing detection for face door system.

Layers:
  1. Blink detection  (weight 0.4) — Eye Aspect Ratio (EAR) via dlib 68 landmarks
  2. Texture analysis  (weight 0.3) — LBP histogram + variance
  3. Optical flow     (weight 0.3) — Farneback flow on face region

Combined score >= 0.5 passes liveness check.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist
from skimage.feature import local_binary_pattern


class LivenessDetector:
    """3-layer anti-spoofing detector using blink, texture, and optical flow."""

    def __init__(self):
        # Thresholds
        self._ear_threshold: float = 0.2       # below this = eye closed
        self._ear_consec_frames: int = 1        # frames closed to count as blink
        self._blink_weight: float = 0.4
        self._texture_weight: float = 0.3
        self._flow_weight: float = 0.3
        self._pass_threshold: float = 0.5

        # LBP params
        self._lbp_radius: int = 1
        self._lbp_points: int = 8

        # Eye landmark indices (68-point model)
        # Left eye:  36-41    Right eye: 42-47
        self._left_eye_idxs = list(range(36, 42))
        self._right_eye_idxs = list(range(42, 48))

        # Load dlib 68-point predictor
        self._predictor = self._load_predictor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_liveness(self, frames: List[np.ndarray]) -> Dict:
        """
        Run all three liveness checks on a sequence of face frames.

        Args:
            frames: List of BGR numpy arrays (at least 2 for flow, ~10+ ideal).

        Returns:
            dict with keys:
                passed: bool
                score: float (0-1 combined)
                blink_score: float (0-1)
                texture_score: float (0-1)
                flow_score: float (0-1)
                details: human-readable summary
        """
        if not frames:
            return {
                "passed": False,
                "score": 0.0,
                "blink_score": 0.0,
                "texture_score": 0.0,
                "flow_score": 0.0,
                "details": "No frames provided",
            }

        # Convert all frames to RGB once (dlib uses RGB)
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        # Layer 1: Blink detection (uses all frames)
        blink_score, blink_detail = self._check_blink(rgb_frames)

        # Layer 2: Texture analysis (uses middle frame)
        texture_score, texture_detail = self._check_texture(frames[len(frames) // 2])

        # Layer 3: Optical flow (needs at least 2 frames)
        flow_score, flow_detail = self._check_optical_flow(gray_frames)

        # Combined score
        score = (
            blink_score * self._blink_weight
            + texture_score * self._texture_weight
            + flow_score * self._flow_weight
        )
        passed = score >= self._pass_threshold

        details = (
            f"Blink({blink_detail}) Texture({texture_detail}) Flow({flow_detail})"
        )

        return {
            "passed": passed,
            "score": round(score, 4),
            "blink_score": round(blink_score, 4),
            "texture_score": round(texture_score, 4),
            "flow_score": round(flow_score, 4),
            "details": details,
        }

    # ------------------------------------------------------------------
    # Layer 1: Blink detection via Eye Aspect Ratio
    # ------------------------------------------------------------------

    def _check_blink(self, rgb_frames: List[np.ndarray]) -> Tuple[float, str]:
        """
        Eye Aspect Ratio (EAR) blink detection.

        EAR = (p2-p6 + p3-p5) / (2 * p1-p4)   for each eye, then averaged.

        A blink is detected when EAR drops below threshold for at least
        `ear_consec_frames` frames and then rises again.
        """
        if len(rgb_frames) < 3:
            return 0.0, "insufficient_frames"

        ears: List[float] = []
        faces_detected = 0

        # We use dlib's frontal face detector once, cache the rect
        detector = dlib.get_frontal_face_detector()
        face_rect: Optional[dlib.rectangle] = None

        for frame in rgb_frames:
            # Detect face if we haven't yet; reuse rect for subsequent frames
            if face_rect is None:
                dets = detector(frame, 0)
                if len(dets) == 0:
                    continue
                face_rect = max(
                    dets, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top())
                )

            # Get 68 landmarks
            shape = self._predictor(frame, face_rect)
            faces_detected += 1

            # Compute EAR for both eyes
            left_ear = self._eye_aspect_ratio(
                [(shape.part(i).x, shape.part(i).y) for i in self._left_eye_idxs]
            )
            right_ear = self._eye_aspect_ratio(
                [(shape.part(i).x, shape.part(i).y) for i in self._right_eye_idxs]
            )
            ear = (left_ear + right_ear) / 2.0
            ears.append(ear)

        if len(ears) < 3:
            return 0.0, "face_not_detected"

        # Count blinks: EAR drops below threshold then rises above
        blink_count = 0
        was_closed = False
        closed_streak = 0

        for ear in ears:
            if ear < self._ear_threshold:
                closed_streak += 1
                was_closed = True
            else:
                if was_closed and closed_streak >= self._ear_consec_frames:
                    blink_count += 1
                was_closed = False
                closed_streak = 0

        # Score: expect at least 1 blink.  Score caps at 1.0
        score = min(1.0, blink_count / 1.0)
        return score, f"blinks={blink_count}"

    @staticmethod
    def _eye_aspect_ratio(eye_points: List[Tuple[int, int]]) -> float:
        """
        Compute Eye Aspect Ratio from 6 landmark points.
        Points: [p1, p2, p3, p4, p5, p6]  (indices match dlib 68-point ordering).
        """
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        if C == 0:
            return 0.0
        return float((A + B) / (2.0 * C))

    # ------------------------------------------------------------------
    # Layer 2: Texture analysis via Local Binary Patterns
    # ------------------------------------------------------------------

    def _check_texture(self, frame: np.ndarray) -> Tuple[float, str]:
        """
        Analyze face texture using LBP histogram + variance.

        Real faces have richer, noisier texture patterns compared to
        printed photos / screen captures (which tend to be smoother or
        have uniform pixelation artifacts). We compute:
          - LBP histogram entropy (higher = more complex texture)
          - Gray-level variance (higher = more natural skin variation)

        Returns a score 0-1 where higher = more likely real.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Use dlib to find face ROI for focused texture analysis
        detector = dlib.get_frontal_face_detector()
        dets = detector(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 0)

        if len(dets) == 0:
            # Fall back to entire frame
            face_roi = gray
        else:
            largest = max(
                dets, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top())
            )
            # Add padding
            h, w = gray.shape
            x1 = max(0, largest.left())
            y1 = max(0, largest.top())
            x2 = min(w, largest.right())
            y2 = min(h, largest.bottom())
            face_roi = gray[y1:y2, x1:x2]

        if face_roi.size < 100:
            return 0.0, "roi_too_small"

        # -- LBP histogram --
        lbp = local_binary_pattern(
            face_roi, self._lbp_points, self._lbp_radius, method="uniform"
        )
        n_bins = self._lbp_points + 2
        hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )

        # Entropy of LBP histogram — real faces have more uniform distribution
        hist = hist + 1e-10  # avoid log(0)
        entropy = -np.sum(hist * np.log2(hist))
        max_entropy = math.log2(n_bins)
        entropy_score = entropy / max_entropy  # 0-1

        # -- Gray-level variance --
        variance = np.var(face_roi.astype(np.float32))
        # Normalize: assume max reasonable variance ~4000 for 8-bit face
        var_score = min(1.0, variance / 4000.0)

        # Combine: both contribute equally
        score = 0.6 * entropy_score + 0.4 * var_score
        score = min(1.0, max(0.0, score))

        return score, f"entropy={entropy:.2f}_var={variance:.0f}"

    # ------------------------------------------------------------------
    # Layer 3: Optical flow analysis
    # ------------------------------------------------------------------

    def _check_optical_flow(self, gray_frames: List[np.ndarray]) -> Tuple[float, str]:
        """
        Farneback optical flow on the face region across consecutive frames.

        A static photo / screen has near-zero optical flow. A real face has
        micro-movements (breathing, slight head motion, etc.) that produce
        measurable flow.

        Returns score 0-1 where higher = more motion = more likely real.
        """
        if len(gray_frames) < 2:
            return 0.0, "insufficient_frames"

        # Detect face in first frame to define ROI
        detector = dlib.get_frontal_face_detector()
        rgb0 = cv2.cvtColor(
            cv2.cvtColor(gray_frames[0], cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB
        )
        dets = detector(rgb0, 0)

        if len(dets) == 0:
            return 0.0, "face_not_detected"

        largest = max(
            dets, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top())
        )
        h, w = gray_frames[0].shape
        x1 = max(0, largest.left())
        y1 = max(0, largest.top())
        x2 = min(w, largest.right())
        y2 = min(h, largest.bottom())

        # Compute flow between consecutive frame pairs within the ROI
        flow_magnitudes: List[float] = []
        prev = gray_frames[0]

        for i in range(1, len(gray_frames)):
            curr = gray_frames[i]

            # Crop ROI from both frames
            prev_roi = prev[y1:y2, x1:x2]
            curr_roi = curr[y1:y2, x1:x2]

            if prev_roi.size < 100 or curr_roi.size < 100:
                continue

            try:
                flow = cv2.calcOpticalFlowFarneback(
                    prev_roi,
                    curr_roi,
                    None,
                    pyr_scale=0.5,
                    levels=3,
                    winsize=15,
                    iterations=3,
                    poly_n=5,
                    poly_sigma=1.2,
                    flags=0,
                )
                mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                mean_mag = float(np.mean(mag))
                flow_magnitudes.append(mean_mag)
            except cv2.error:
                continue

            prev = curr

        if len(flow_magnitudes) == 0:
            return 0.0, "flow_compute_error"

        avg_flow = np.mean(flow_magnitudes)
        max_flow = np.max(flow_magnitudes)

        # Real faces: avg_flow typically 0.1–2.0+ pixels
        # Static photo: avg_flow << 0.1
        # Score uses a sigmoid-like mapping
        score = min(1.0, avg_flow / 0.8)
        score = min(1.0, max(0.0, score))

        return score, f"avg_flow={avg_flow:.3f}_max={max_flow:.3f}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_predictor() -> dlib.shape_predictor:
        """Load the 68-point face landmark predictor."""
        import os

        paths = [
            "/usr/share/dlib/shape_predictor_68_face_landmarks.dat",
            os.path.join(
                os.path.dirname(os.path.dirname(dlib.__file__)),
                "data",
                "shape_predictor_68_face_landmarks.dat",
            ),
        ]

        for p in paths:
            if os.path.exists(p):
                return dlib.shape_predictor(p)

        raise FileNotFoundError(
            "Could not find shape_predictor_68_face_landmarks.dat. "
            "Install via: sudo apt install dlib-data  or place it in /usr/share/dlib/"
        )
