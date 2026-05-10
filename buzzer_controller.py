"""
buzzer_controller.py — 5V active buzzer via GPIO.

Provides BuzzerController class for audible feedback (success/fail/generic
beep patterns) via a PWM-capable active buzzer on a single GPIO pin.
Uses BCM numbering.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_GPIO: Optional[type] = None


def _get_gpio():
    """Import and return RPi.GPIO module, or None if unavailable."""
    global _GPIO
    if _GPIO is None:
        try:
            import RPi.GPIO as GPIO

            _GPIO = GPIO
        except (ImportError, RuntimeError) as exc:
            logger.warning("RPi.GPIO not available: %s", exc)
            return None
    return _GPIO


class BuzzerController:
    """Controls a 5V active buzzer via a single GPIO pin."""

    def __init__(self, pin: int = 18):
        """
        Set up the GPIO pin as an output, initialized LOW (off).

        Args:
            pin: BCM GPIO pin number (default 18).
        """
        self.pin = pin
        self._gpio = _get_gpio()
        if self._gpio is not None:
            try:
                self._gpio.setmode(self._gpio.BCM)
                self._gpio.setwarnings(False)
                self._gpio.setup(self.pin, self._gpio.OUT, initial=self._gpio.LOW)
                logger.info("Buzzer pin %d initialized LOW (off)", self.pin)
            except Exception as exc:
                logger.error("Failed to initialize buzzer on pin %d: %s", pin, exc)
                self._gpio = None
        else:
            logger.warning("RPi.GPIO unavailable — buzzer operations will be no-ops")

    # ------------------------------------------------------------------
    # High-level patterns
    # ------------------------------------------------------------------

    def success_beep(self) -> bool:
        """
        Single short beep indicating successful face recognition.

        Pattern: 0.1 s ON, 0.05 s OFF.
        """
        return self.beep(times=1, on_time=0.1, off_time=0.05)

    def fail_beep(self) -> bool:
        """
        Three rapid beeps indicating failed face recognition.

        Pattern: 0.15 s ON, 0.1 s OFF, repeated 3 times.
        """
        return self.beep(times=3, on_time=0.15, off_time=0.1)

    # ------------------------------------------------------------------
    # Generic beep pattern
    # ------------------------------------------------------------------

    def beep(
        self, times: int = 1, on_time: float = 0.1, off_time: float = 0.1
    ) -> bool:
        """
        Generate a beep pattern.

        Args:
            times:    Number of beeps in the sequence.
            on_time:  Seconds the buzzer is ON (HIGH) per beep.
            off_time: Seconds the buzzer is OFF (LOW) between beeps.

        Returns:
            True if the pattern completed, False on error.
        """
        if self._gpio is None:
            logger.warning("GPIO not available — beep is a no-op")
            return False

        try:
            for i in range(times):
                self._gpio.output(self.pin, self._gpio.HIGH)
                time.sleep(on_time)
                self._gpio.output(self.pin, self._gpio.LOW)
                if i < times - 1:
                    time.sleep(off_time)
            return True
        except Exception as exc:
            logger.error("Buzzer beep failed on pin %d: %s", self.pin, exc)
            # Best-effort: ensure buzzer is off
            try:
                self._gpio.output(self.pin, self._gpio.LOW)
            except Exception:
                pass
            return False

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """Clean up GPIO resources (call on shutdown)."""
        if self._gpio is None:
            return
        try:
            self._gpio.cleanup(self.pin)
            logger.info("GPIO cleanup for buzzer pin %d", self.pin)
        except Exception as exc:
            logger.warning("GPIO cleanup warning on pin %d: %s", self.pin, exc)
