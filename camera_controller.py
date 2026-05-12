"""camera_controller.py — Pi Camera v2 control via picamera2.

Provides CameraController class for capturing frames in BGR format (OpenCV-compatible).
Supports context manager, single frame, and burst capture at ~30fps.
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class CameraController:
    """Controls Pi Camera v2 via picamera2, returning BGR frames for OpenCV."""

    def __init__(self, resolution: tuple = (640, 480), framerate: int = 30):
        """
        Initialize picamera2 with given resolution and framerate.

        Args:
            resolution: (width, height) tuple in pixels.
            framerate: Target frames per second.
        """
        self.resolution = resolution
        self.framerate = framerate
        self._camera = None
        # High-res config for liveness capture (full sensor detail)
        self._liveness_res = (1640, 1232)
        self._liveness_config = None

    def start(self) -> bool:
        """Start the camera with configured resolution and framerate."""
        try:
            from picamera2 import Picamera2
            from libcamera import Transform

            self._camera = Picamera2()

            # Build video config for best performance
            config = self._camera.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={"FrameRate": self.framerate},
                transform=Transform(),
            )
            self._camera.configure(config)
            self._camera.start()
            logger.info(
                "Camera started: %s @ %d fps", self.resolution, self.framerate
            )
            return True
        except Exception as exc:
            logger.error("Failed to start camera: %s", exc)
            self._camera = None
            return False

    def stop(self) -> None:
        """Stop the camera and release resources."""
        if self._camera is not None:
            try:
                self._camera.stop()
                self._camera.close()
            except Exception as exc:
                logger.warning("Error stopping camera: %s", exc)
            finally:
                self._camera = None

    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame.

        Returns:
            np.ndarray in BGR format (H x W x 3), or None on failure.
        """
        if self._camera is None:
            logger.warning("Camera not started — call start() first.")
            return None

        try:
            # picamera2 returns RGB888 — convert to BGR for OpenCV
            rgb = self._camera.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            return bgr
        except Exception as exc:
            logger.error("Failed to capture frame: %s", exc)
            return None

    def capture_liveness_frame(self) -> Optional[np.ndarray]:
        """Capture a single high-res frame (1640x1232) for texture liveness.

        Temporarily reconfigures camera to full sensor detail mode,
        captures one frame, then restores normal 640x480 mode.
        Takes ~0.5s for the reconfiguration.

        Returns:
            np.ndarray in BGR format (1232x1640x3), or None on failure.
        """
        if self._camera is None:
            logger.warning("Camera not started")
            return None

        try:
            from libcamera import Transform

            # ── Switch to high-res ──
            self._camera.stop()
            hires = self._camera.create_video_configuration(
                main={"size": self._liveness_res, "format": "RGB888"},
                controls={"FrameRate": 10},
                transform=Transform(),
            )
            self._camera.configure(hires)
            self._camera.start()
            # Wait for AEC/AGC to settle at new resolution
            time.sleep(0.3)

            rgb = self._camera.capture_array()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            logger.info("Captured liveness frame at %s", self._liveness_res)

            # ── Restore normal resolution ──
            self._camera.stop()
            normal = self._camera.create_video_configuration(
                main={"size": self.resolution, "format": "RGB888"},
                controls={"FrameRate": self.framerate},
                transform=Transform(),
            )
            self._camera.configure(normal)
            self._camera.start()

            return bgr

        except Exception as exc:
            logger.error("Failed to capture liveness frame: %s", exc)
            # Attempt to restore normal mode
            try:
                self._camera.stop()
                from libcamera import Transform
                normal = self._camera.create_video_configuration(
                    main={"size": self.resolution, "format": "RGB888"},
                    controls={"FrameRate": self.framerate},
                    transform=Transform(),
                )
                self._camera.configure(normal)
                self._camera.start()
            except Exception:
                pass
            return None

    def capture_frames(self, count: int = 30) -> List[np.ndarray]:
        """
        Capture multiple consecutive frames at ~30fps.

        Args:
            count: Number of frames to capture.

        Returns:
            List of BGR np.ndarray frames. May be shorter than `count` on error.
        """
        if self._camera is None:
            logger.warning("Camera not started — call start() first.")
            return []

        frames: List[np.ndarray] = []
        for i in range(count):
            try:
                rgb = self._camera.capture_array()
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                frames.append(bgr)
            except Exception as exc:
                logger.error("Failed to capture frame %d/%d: %s", i + 1, count, exc)
                break  # stop early on persistent error
        return frames

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraController":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[type],
        exc_val: Optional[BaseException],
        exc_tb: Optional[object],
    ) -> None:
        self.stop()
