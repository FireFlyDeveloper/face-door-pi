"""
relay_controller.py — 5V relay module for solenoid lock via GPIO.

Provides RelayController class to unlock (energize relay) and lock
(de-energize relay) a solenoid door strike on a configurable GPIO pin.
Uses BCM numbering.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy import of RPi.GPIO so that this module can be imported on non-RPi
# systems for documentation / testing stubs.
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


class RelayController:
    """Controls a 5V relay module (solenoid lock) via a single GPIO pin."""

    def __init__(self, pin: int = 17):
        """
        Set up the GPIO pin as an output, initialized LOW (locked).

        Args:
            pin: BCM GPIO pin number (default 17).
        """
        self.pin = pin
        self._gpio = _get_gpio()
        if self._gpio is not None:
            try:
                self._gpio.setmode(self._gpio.BCM)
                self._gpio.setwarnings(False)
                self._gpio.setup(self.pin, self._gpio.OUT, initial=self._gpio.LOW)
                logger.info("Relay pin %d initialized LOW (locked)", self.pin)
            except Exception as exc:
                logger.error("Failed to initialize relay on pin %d: %s", pin, exc)
                self._gpio = None
        else:
            logger.warning("RPi.GPIO unavailable — relay operations will be no-ops")

    def unlock(self, duration: float = 3.0) -> bool:
        """
        Energize the relay for *duration* seconds (door unlocked), then lock.

        Args:
            duration: Seconds to hold the relay energized (default 3.0).

        Returns:
            True if the operation completed successfully, False on error.
        """
        if self._gpio is None:
            logger.warning("GPIO not available — unlock is a no-op")
            return False

        try:
            logger.info("Unlocking door (pin %d HIGH for %.1f s)", self.pin, duration)
            self._gpio.output(self.pin, self._gpio.HIGH)
            time.sleep(duration)
            self._gpio.output(self.pin, self._gpio.LOW)
            logger.info("Door locked (pin %d LOW)", self.pin)
            return True
        except Exception as exc:
            logger.error("Relay unlock failed on pin %d: %s", self.pin, exc)
            # Best-effort restore to safe state
            try:
                self._gpio.output(self.pin, self._gpio.LOW)
            except Exception:
                pass
            return False

    def lock(self) -> None:
        """
        Immediately de-energize the relay (door locked / safe state).
        """
        if self._gpio is None:
            return
        try:
            self._gpio.output(self.pin, self._gpio.LOW)
            logger.info("Door locked (pin %d LOW)", self.pin)
        except Exception as exc:
            logger.error("Failed to lock relay on pin %d: %s", self.pin, exc)

    def cleanup(self) -> None:
        """Clean up GPIO resources (call on shutdown)."""
        if self._gpio is None:
            return
        try:
            self._gpio.cleanup(self.pin)
            logger.info("GPIO cleanup for relay pin %d", self.pin)
        except Exception as exc:
            logger.warning("GPIO cleanup warning on pin %d: %s", self.pin, exc)
