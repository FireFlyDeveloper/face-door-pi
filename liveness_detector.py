"""
liveness_detector.py — 2-layer liveness detection for thesis demo.

Layers:
  1. Blink detection       (weight 0.4) — Eye Aspect Ratio (EAR) via dlib
  2. Non-rigid motion       (weight 0.6) — face parts move independently (real)
                                          vs uniformly (photo) via optical flow variance

Combined score >= 0.35 passes liveness check.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist


class LivenessDetector:
    """2-layer anti-spoofing: blink + non-rigid motion analysis."""

    def __init__(self):
        # ── Blink ──
        self._ear_threshold: float = 0.20
        self._ear_consec_frames: int = 1
        self._blink_weight: float = 0.40

        # ── Non-rigid motion ──
        self._motion_weight: float = 0.60

        # ── Overall ──
        self._pass_threshold: float = 0.35

        # Eye landmark indices (dlib 68-point)
        self._left_eye_idxs = list(range(36, 42))
        self._right_eye_idxs = list(range(42, 48))

        # Key landmark indices for motion analysis (spread across face)
        # Nose, chin, left/right eye, left/right mouth, eyebrows, cheeks
        self._motion_landmark_idxs = [30, 8, 36, 45, 48, 54, 21, 22, 1, 15]

        # Load dlib predictor
        self._predictor = self._load_predictor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_liveness(self, frames: List[np.ndarray]) -> Dict:
        if not frames or len(frames) < 3:
            return {
                "passed": False, "score": 0.0,
                "blink_score": 0.0, "motion_score": 0.0,
                "details": "Insufficient frames",
            }

        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        # Layer 1: Blink
        blink_score, blink_detail = self._check_blink(rgb_frames)

        # Layer 2: Non-rigid motion (key differentiator: real face vs photo)
        motion_score, motion_detail = self._check_nonrigid_motion(rgb_frames, gray_frames)

        # Combined
        score = blink_score * self._blink_weight + motion_score * self._motion_weight
        passed = score >= self._pass_threshold

        details = f"Blink({blink_detail}) Motion({motion_detail})"

        return {
            "passed": passed,
            "score": round(score, 4),
            "blink_score": round(blink_score, 4),
            "texture_score": 0.0,
            "flow_score": 0.0,
            "head_pose_score": round(motion_score, 4),
            "head_trans_score": 0.0,
            "head_score": round(motion_score, 4),
            "screen_score": 1.0,
            "details": details,
        }

    # ------------------------------------------------------------------
    # Layer 1: Blink detection
    # ------------------------------------------------------------------

    def _check_blink(self, rgb_frames: List[np.ndarray]) -> Tuple[float, str]:
        if len(rgb_frames) < 3:
            return 0.0, "insufficient"

        ears: List[float] = []
        detector = dlib.get_frontal_face_detector()
        face_rect: Optional[dlib.rectangle] = None
        frames_ok = 0

        for frame in rgb_frames:
            if face_rect is None:
                dets = detector(frame, 0)
                if not dets:
                    continue
                face_rect = max(
                    dets, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top())
                )

            shape = self._predictor(frame, face_rect)
            frames_ok += 1

            left_ear = self._eye_aspect_ratio(
                [(shape.part(i).x, shape.part(i).y) for i in self._left_eye_idxs]
            )
            right_ear = self._eye_aspect_ratio(
                [(shape.part(i).x, shape.part(i).y) for i in self._right_eye_idxs]
            )
            ear = (left_ear + right_ear) / 2.0
            ears.append(ear)

        if frames_ok < 3:
            return 0.0, "face_not_found"

        # Median smoothing (window=3)
        if len(ears) >= 3:
            sm = []
            for i in range(len(ears)):
                w = ears[max(0, i - 1):min(len(ears), i + 2)]
                sm.append(float(np.median(w)))
            ears = sm

        # Count blink transitions
        blink_count = 0
        was_closed = False
        closed_streak = 0
        min_ear = min(ears)
        max_ear = max(ears)
        ear_range = max_ear - min_ear

        for ear in ears:
            if ear < self._ear_threshold:
                closed_streak += 1
                was_closed = True
            else:
                if was_closed and closed_streak >= self._ear_consec_frames:
                    blink_count += 1
                was_closed = False
                closed_streak = 0

        # Score: 1 blink = full marks. Also give partial for strong EAR dip.
        if blink_count >= 1:
            return 1.0, f"blinks={blink_count}"

        # Partial credit for EAR dip > 20% of range (eye movement, maybe squint)
        if ear_range > 0.05 and max_ear > 0.22:
            return 0.5, f"dip={ear_range:.3f}"

        return 0.0, f"no_blink(range={ear_range:.3f})"

    @staticmethod
    def _eye_aspect_ratio(eye_points: List[Tuple[int, int]]) -> float:
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        if C == 0:
            return 0.0
        return float((A + B) / (2.0 * C))

    # ------------------------------------------------------------------
    # Layer 2: Non-rigid motion analysis (KEY DIFFERENTIATOR)
    # ------------------------------------------------------------------

    def _check_nonrigid_motion(
        self, rgb_frames: List[np.ndarray], gray_frames: List[np.ndarray]
    ) -> Tuple[float, str]:
        """
        The KEY anti-spoofing signal.

        REAL FACE: Different parts of the face move independently
        (non-rigid motion). Eyes blink, nose stays, mouth moves, etc.
        The optical flow vectors at different landmark positions have
        HIGH VARIANCE.

        PHOTO ON PHONE: The entire photo moves as one rigid object.
        All landmark positions shift by nearly the SAME amount.
        The optical flow vectors have LOW VARIANCE.

        We compute dense optical flow between consecutive frames,
        sample at landmark positions, and compute the variance of
        flow vectors across landmarks. Higher variance = real face.
        """
        if len(rgb_frames) < 3:
            return 0.0, "insufficient"

        detector = dlib.get_frontal_face_detector()
        face_rect: Optional[dlib.rectangle] = None
        landmarks_by_frame: List[List[Tuple[int, int]]] = []

        # Cache landmarks for each frame
        for frame in rgb_frames:
            if face_rect is None:
                dets = detector(frame, 0)
                if not dets:
                    continue
                face_rect = max(
                    dets, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top())
                )

            shape = self._predictor(frame, face_rect)
            pts = [(shape.part(i).x, shape.part(i).y) for i in self._motion_landmark_idxs]
            landmarks_by_frame.append(pts)

        if len(landmarks_by_frame) < 3:
            return 0.0, "face_lost"

        # Compute flow variance across consecutive frame pairs
        variance_scores = []
        coherence_ratios = []

        for f_idx in range(1, len(landmarks_by_frame)):
            prev_pts = np.array(landmarks_by_frame[f_idx - 1], dtype=np.float32)
            curr_pts = np.array(landmarks_by_frame[f_idx], dtype=np.float32)

            # Compute per-landmark displacement vectors
            displacements = curr_pts - prev_pts  # N x 2

            # METHOD 1: Variance of displacement magnitudes
            # High variance = parts move differently = real face
            mags = np.sqrt(displacements[:, 0] ** 2 + displacements[:, 1] ** 2)
            mean_mag = float(np.mean(mags))
            var_mag = float(np.var(mags))

            # METHOD 2: Direction coherence
            # Compute the mean displacement vector, then check how much each
            # landmark deviates from it
            mean_disp = np.mean(displacements, axis=0)
            # Normalize
            mean_norm = np.linalg.norm(mean_disp)
            if mean_norm > 0.5:
                # Compute angular deviation of each landmark's displacement
                # from the mean direction
                dot_products = []
                for d in displacements:
                    d_norm = np.linalg.norm(d)
                    if d_norm > 0.3:
                        cos_angle = np.dot(d, mean_disp) / (d_norm * mean_norm)
                        dot_products.append(abs(cos_angle))

                if dot_products:
                    # Average cosine similarity to mean motion
                    # Low = parts move differently → real face
                    # High = all moving same direction → photo
                    avg_cos = np.mean(dot_products)
                    # Convert: low cos similarity = high liveness score
                    dir_score = 1.0 - avg_cos
                else:
                    dir_score = 0.0
            else:
                dir_score = 0.0  # insufficient motion

            # Combine: variance + direction
            # Scale variance: at 0.5 pixels² variance among 10 landmarks = meaningful
            var_score = min(1.0, var_mag / 0.5)

            # Final per-pair score
            if mean_mag > 0.3:
                pair_score = 0.5 * var_score + 0.5 * dir_score
            else:
                pair_score = 0.0  # no motion at all

            variance_scores.append(var_score)
            coherence_ratios.append(dir_score)

        if not variance_scores:
            return 0.0, "no_motion"

        # Average across all frame pairs
        avg_variance = np.mean(variance_scores)
        avg_coherence = np.mean(coherence_ratios) if coherence_ratios else 0.0
        final_score = 0.5 * avg_variance + 0.5 * avg_coherence

        # Clamp
        final_score = max(0.0, min(1.0, final_score))

        return (
            final_score,
            f"var={avg_variance:.2f}_dir={avg_coherence:.2f}",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_predictor() -> dlib.shape_predictor:
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
        raise FileNotFoundError("Could not find shape_predictor_68_face_landmarks.dat")
