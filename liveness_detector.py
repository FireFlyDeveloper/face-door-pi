"""
liveness_detector.py — Reliable 2-layer liveness detection for thesis demo.

Layers:
  1. Blink detection  (weight 0.5) — Eye Aspect Ratio (EAR) via dlib 68 landmarks
  2. Head pose motion (weight 0.5) — 3D head pose estimation via PnP + dlib landmarks

Combined score >= 0.4 passes liveness check.

A static photo / phone screen fails both checks:
  - No blink: EAR stays constant (no transition below threshold)
  - No 3D head motion: solvePnP rotation stays flat

A real person passes both after 2.5s of capture.
"""

import math
from typing import Dict, List, Optional, Tuple

import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist


class LivenessDetector:
    """2-layer anti-spoofing: blink detection + head pose motion."""

    def __init__(self):
        # ── Blink thresholds ──
        self._ear_threshold: float = 0.22       # EAR below this = eye closed
        self._ear_consec_frames: int = 1         # frames closed to count as blink
        self._eye_close_max_ear: float = 0.20    # typical max EAR when truly closed
        self._blink_weight: float = 0.50

        # ── Head pose thresholds ──
        self._head_weight: float = 0.50
        self._head_motion_threshold: float = 1.5  # degrees of total rotation change
        self._head_trans_threshold: float = 3.0   # pixels of translation change

        # ── Overall ──
        self._pass_threshold: float = 0.40

        # ── 3D face model points (generic) for solvePnP ──
        #   Nose tip, chin, left eye corner, right eye corner,
        #   left mouth corner, right mouth corner
        self._model_points = np.array([
            (0.0,    0.0,    0.0),       # Nose tip
            (0.0,   -6.0,    0.0),       # Chin
            (-2.5,   2.0,   -2.0),       # Left eye left corner
            (2.5,    2.0,   -2.0),       # Right eye right corner
            (-2.0,   4.0,   -2.0),       # Left mouth corner
            (2.0,    4.0,   -2.0),       # Right mouth corner
        ], dtype=np.float64)

        # Camera intrinsic matrix (assumes 640x480, ~60° FOV)
        self._camera_matrix = np.array([
            [520.0,   0.0,   320.0],
            [  0.0,  520.0,  240.0],
            [  0.0,    0.0,    1.0],
        ], dtype=np.float64)
        self._dist_coeffs = np.zeros((4, 1))

        # Landmark indices for the 6 key points above (dlib 68-point model)
        self._nose_idx = 30
        self._chin_idx = 8
        self._left_eye_idx = 36
        self._right_eye_idx = 45
        self._left_mouth_idx = 48
        self._right_mouth_idx = 54

        # Eye landmark indices
        self._left_eye_idxs = list(range(36, 42))
        self._right_eye_idxs = list(range(42, 48))

        # Load dlib predictor
        self._predictor = self._load_predictor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_liveness(self, frames: List[np.ndarray]) -> Dict:
        """
        Run blink + head-pose liveness checks.

        Args:
            frames: List of BGR numpy arrays (30 frames @ ~0.08s each = 2.5s).

        Returns:
            dict with keys: passed, score, blink_score, head_score, details
        """
        if not frames or len(frames) < 3:
            return {
                "passed": False, "score": 0.0,
                "blink_score": 0.0, "head_score": 0.0,
                "head_pose_score": 0.0, "head_trans_score": 0.0,
                "details": "Insufficient frames",
            }

        # Convert frames once
        rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
        gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

        # Layer 1: Blink
        blink_score, blink_detail = self._check_blink(rgb_frames)

        # Layer 2: Head pose motion
        head_pose_score, head_trans_score, head_detail = self._check_head_pose(
            rgb_frames, gray_frames
        )

        # Layer 3: Screen detection (phone vs real face texture)
        screen_score, screen_detail = self._check_screen(rgb_frames)

        # Combined head score: rotation ONLY — translation is easy to fake (phone movement)
        head_score = head_pose_score

        # Total = weighted sum
        score = blink_score * self._blink_weight + head_score * self._head_weight

        # HARD RULE: A static picture CANNOT blink. If no blink and no near-blink,
        # hard-cap the score below threshold regardless of other layers.
        if blink_score < 0.3:
            score = min(score, self._pass_threshold - 0.05)

        # HARD RULE: If screen is detected (phone/tablet), reduce score heavily
        if screen_score < 0.3:
            score *= 0.3

        passed = score >= self._pass_threshold

        details = (
            f"Blink({blink_detail}) "
            f"Head({head_detail}) "
            f"Screen({screen_detail})"
        )

        return {
            "passed": passed,
            "score": round(score, 4),
            "blink_score": round(blink_score, 4),
            "texture_score": 0.0,
            "flow_score": 0.0,
            "head_pose_score": round(head_pose_score, 4),
            "head_trans_score": round(head_trans_score, 4),
            "head_score": round(head_score, 4),
            "screen_score": round(screen_score, 4),
            "details": details,
        }

    # ------------------------------------------------------------------
    # Layer 1: Blink detection
    # ------------------------------------------------------------------

    def _check_blink(self, rgb_frames: List[np.ndarray]) -> Tuple[float, str]:
        """
        Eye Aspect Ratio (EAR) blink detection with smoothing.

        A blink is counted when EAR drops below threshold then rises above.
        We use a rolling median of 3 EAR values to reduce noise.
        """
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

        # Smooth with rolling median (window=3)
        if len(ears) >= 3:
            smoothed = []
            for i in range(len(ears)):
                window = ears[max(0, i - 1):min(len(ears), i + 2)]
                smoothed.append(float(np.median(window)))
            ears = smoothed

        # Count blinks
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

        # If no blink detected, check if eyes were always "low" (might be
        # squinting or poor landmarks) — still give partial score if EAR
        # dropped at least 15% from max
        if blink_count == 0 and len(ears) >= 3:
            max_ear = max(ears)
            min_ear = min(ears)
            dip_ratio = (max_ear - min_ear) / max_ear if max_ear > 0 else 0
            if dip_ratio > 0.15:
                blink_score = 0.6  # partial credit for eye movement
                return blink_score, f"near_blink(dip={dip_ratio:.2f})"
            return 0.1, f"no_blink(dip={dip_ratio:.2f})"

        score = min(1.0, blink_count / 1.0)
        return score, f"blinks={blink_count}"

    @staticmethod
    def _eye_aspect_ratio(eye_points: List[Tuple[int, int]]) -> float:
        A = dist.euclidean(eye_points[1], eye_points[5])
        B = dist.euclidean(eye_points[2], eye_points[4])
        C = dist.euclidean(eye_points[0], eye_points[3])
        if C == 0:
            return 0.0
        return float((A + B) / (2.0 * C))

    # ------------------------------------------------------------------
    # Layer 2: Head pose motion (3D rotation + translation)
    # ------------------------------------------------------------------

    def _check_head_pose(
        self, rgb_frames: List[np.ndarray], gray_frames: List[np.ndarray]
    ) -> Tuple[float, float, str]:
        """
        Estimate head pose (rotation + translation) via PnP.

        A real person shows:
          - Rotation variation (yaw/pitch/roll changes of 2-8°)
          - Translation variation (head moves in 3D space)

        A static photo shows:
          - Near-zero rotation and translation variation (< 0.5°)
        """
        if len(rgb_frames) < 3:
            return 0.0, 0.0, "insufficient"

        detector = dlib.get_frontal_face_detector()
        face_rect: Optional[dlib.rectangle] = None

        # Collect rotation vectors for each frame where face is found
        rvecs: List[np.ndarray] = []
        tvecs: List[np.ndarray] = []
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

            # Get 6 key 2D image points
            image_points = np.array([
                (shape.part(self._nose_idx).x,      shape.part(self._nose_idx).y),
                (shape.part(self._chin_idx).x,      shape.part(self._chin_idx).y),
                (shape.part(self._left_eye_idx).x,  shape.part(self._left_eye_idx).y),
                (shape.part(self._right_eye_idx).x, shape.part(self._right_eye_idx).y),
                (shape.part(self._left_mouth_idx).x, shape.part(self._left_mouth_idx).y),
                (shape.part(self._right_mouth_idx).x, shape.part(self._right_mouth_idx).y),
            ], dtype=np.float64)

            try:
                success, rvec, tvec = cv2.solvePnP(
                    self._model_points, image_points,
                    self._camera_matrix, self._dist_coeffs,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if success:
                    rvecs.append(rvec.flatten())
                    tvecs.append(tvec.flatten())
            except cv2.error:
                continue

        if len(rvecs) < 3:
            return 0.0, 0.0, f"face_lost({frames_ok}frames)"

        # Compute std dev of rotation (yaw/pitch/roll) across frames
        rvec_arr = np.array(rvecs)  # N x 3
        tvec_arr = np.array(tvecs)  # N x 3

        # Convert rotation vectors to Euler angles for interpretable measurement
        eulers = []
        for rv in rvecs:
            rot_mat, _ = cv2.Rodrigues(rv.reshape(3, 1))
            sy = math.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                x = math.atan2(rot_mat[2, 1], rot_mat[2, 2])
                y = math.atan2(-rot_mat[2, 0], sy)
                z = math.atan2(rot_mat[1, 0], rot_mat[0, 0])
            else:
                x = math.atan2(-rot_mat[1, 2], rot_mat[1, 1])
                y = math.atan2(-rot_mat[2, 0], sy)
                z = 0
            eulers.append([math.degrees(x), math.degrees(y), math.degrees(z)])
        euler_arr = np.array(eulers)

        # Rotation variation: max-min range across all 3 axes
        rot_range = np.max(euler_arr, axis=0) - np.min(euler_arr, axis=0)
        total_rot_change = float(np.linalg.norm(rot_range))

        # Translation variation: max-min range
        trans_range = np.max(tvec_arr, axis=0) - np.min(tvec_arr, axis=0)
        total_trans_change = float(np.linalg.norm(trans_range))

        # Score rotation: scale so 3° change = full score
        pose_score = min(1.0, total_rot_change / 3.0)

        # Score translation: scale so 10px change = full score
        trans_score = min(1.0, total_trans_change / 10.0)

        return (
            pose_score,
            trans_score,
            f"rot={total_rot_change:.1f}°_trans={total_trans_change:.0f}px",
        )

    # ------------------------------------------------------------------
    # Layer 3: Screen detection (phone/tablet anti-spoofing)
    # ------------------------------------------------------------------

    def _check_screen(self, rgb_frames: List[np.ndarray]) -> Tuple[float, str]:
        """
        Detect if the face is displayed on a screen (phone/tablet) vs real.

        Uses two signals:
          1. Laplacian variance: phone screens have unnaturally uniform
             texture compared to real skin (lower variance).
          2. Edge density: phone screens have sharp rectangular borders
             (the device bezel) visible in the frame.

        Returns score 0-1 where higher = more likely a real face.
        """
        if not rgb_frames:
            return 0.5, "no_frames"

        # Use the middle frame
        frame = rgb_frames[len(rgb_frames) // 2]
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        # ── Signal 1: Laplacian variance (texture uniformity) ──
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(np.var(lap))

        # Real faces at 640x480 typically have lap_var 20-200+
        # Phone screens showing a face typically have lap_var 5-40
        # (smoother due to pixel grid + display backlight)
        # Score: higher variance = more likely real
        if lap_var < 10:
            lap_score = 0.0  # very uniform → screen
        elif lap_var > 60:
            lap_score = 1.0  # rich texture → real face
        else:
            lap_score = (lap_var - 10) / 50.0  # linear ramp

        lap_score = max(0.0, min(1.0, lap_score))

        # ── Signal 2: Edge density in peripheral regions ──
        # Phone screens have sharp bezel edges. We look for straight
        # edges near the frame borders using Hough lines.
        edges = cv2.Canny(gray, 50, 150)
        # Only check the outer 20% of the frame (screen bezel area)
        border_w = int(w * 0.2)
        border_h = int(h * 0.2)
        # Create a mask for the border region (ring around center)
        mask = np.zeros((h, w), dtype=np.uint8)
        # Fill the outer ring
        mask[border_h:-border_h, border_w:-border_w] = 255
        mask = cv2.bitwise_not(mask)
        border_edges = cv2.bitwise_and(edges, edges, mask=mask)

        # Count strong edge pixels in the border region
        border_edge_count = np.count_nonzero(border_edges)
        total_border_pixels = np.count_nonzero(mask)
        border_edge_density = border_edge_count / total_border_pixels if total_border_pixels > 0 else 0

        # Real faces: low edge density in peripheral area
        # Phone screen: higher edge density (bezel edges)
        if border_edge_density < 0.005:
            border_score = 1.0  # natural background
        elif border_edge_density > 0.03:
            border_score = 0.0  # lots of edges → possible screen border
        else:
            border_score = 1.0 - (border_edge_density - 0.005) / 0.025

        border_score = max(0.0, min(1.0, border_score))

        # ── Combined score ──
        score = 0.6 * lap_score + 0.4 * border_score
        score = max(0.0, min(1.0, score))

        return score, f"lapvar={lap_var:.0f}_edge={border_edge_density:.4f}"

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
            "Could not find shape_predictor_68_face_landmarks.dat"
        )
