"""
ir_led_controller.py — GPIO control for IR LED illuminator.

Pi NoIR camera has no IR filter, so IR LEDs are used for:
  - Night vision in low light
  - IR liveness detection (alternating IR on/off frames)

Uses BCM numbering. RPi.GPIO is imported lazily for testing on non-RPi systems.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_GPIO: Optional[type] = None


def _get_gpio():
    global _GPIO
    if _GPIO is None:
        try:
            import RPi.GPIO as GPIO
            _GPIO = GPIO
        except (ImportError, RuntimeError) as exc:
            logger.warning("RPi.GPIO not available: %s", exc)
            return None
    return _GPIO


class IRLEDController:
    """Controls an IR LED illuminator array via a single GPIO pin.

    Typical circuit: GPIO → NPN transistor (2N2222) base → drives IR LEDs.
    GPIO LOW = LEDs OFF, GPIO HIGH = LEDs ON.
    """

    def __init__(self, pin: int = 23):
        """Initialize IR LED controller.

        Args:
            pin: BCM GPIO pin number driving the IR LED transistor (default 23).
        """
        self.pin = pin
        self._gpio = _get_gpio()
        self._is_on = False

        if self._gpio is not None:
            try:
                self._gpio.setmode(self._gpio.BCM)
                self._gpio.setwarnings(False)
                self._gpio.setup(self.pin, self._gpio.OUT, initial=self._gpio.LOW)
                logger.info("IR LED pin %d initialized LOW (off)", self.pin)
            except Exception as exc:
                logger.error("Failed to init IR LED pin %d: %s", pin, exc)
                self._gpio = None
        else:
            logger.warning("RPi.GPIO unavailable — IR LED operations are no-ops")

    def on(self) -> bool:
        """Turn IR LEDs on."""
        if self._gpio is None:
            return False
        try:
            self._gpio.output(self.pin, self._gpio.HIGH)
            self._is_on = True
            return True
        except Exception as exc:
            logger.error("Failed to turn IR LEDs ON: %s", exc)
            return False

    def off(self) -> bool:
        """Turn IR LEDs off."""
        if self._gpio is None:
            return False
        try:
            self._gpio.output(self.pin, self._gpio.LOW)
            self._is_on = False
            return True
        except Exception as exc:
            logger.error("Failed to turn IR LEDs OFF: %s", exc)
            return False

    @property
    def is_on(self) -> bool:
        """True if IR LEDs are currently on."""
        return self._is_on

    def capture_pair(self, camera, settle_time: float = 0.05) -> tuple:
        """Capture two frames: one with IR ON, one with IR OFF.

        Args:
            camera: CameraController instance (must have capture_frame()).
            settle_time: Seconds to wait after toggling for LEDs to stabilize.

        Returns:
            (ir_on_frame, ir_off_frame) as BGR numpy arrays, or (None, None) on failure.
        """
        # Capture with IR ON
        self.on()
        time.sleep(settle_time)
        ir_on = camera.capture_frame()

        # Capture with IR OFF
        self.off()
        time.sleep(settle_time)
        ir_off = camera.capture_frame()

        if ir_on is None or ir_off is None:
            logger.warning("IR capture pair incomplete — IR frames may be None")
            return None, None

        return ir_on, ir_off

    def cleanup(self) -> None:
        """Turn off IR LEDs and release GPIO."""
        if self._gpio is None:
            return
        try:
            self.off()
            self._gpio.cleanup(self.pin)
            logger.info("GPIO cleanup for IR LED pin %d", self.pin)
        except Exception as exc:
            logger.warning("GPIO cleanup warning on pin %d: %s", self.pin, exc)
