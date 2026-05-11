"""
relay_controller.py — 5V relay module for solenoid lock via GPIO.

Active-low relay: LOW = ON (energized/unlocked), HIGH = OFF (de-energized/locked).

Provides:
  - unlock(duration)  — pulse unlock for N seconds (face-recognition auto-unlock)
  - lock() / unlock_persistent()  — set persistent relay state (RF remote / manual)
  - set_state(locked)  — unified persistent setter

Uses BCM numbering. RPi.GPIO is imported lazily so this module can be
imported for testing on non-RPi systems.
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


class RelayController:
    """Controls a 5V relay module for door lock actuation.

    Active-low module: GPIO LOW = relay ON (unlocked), HIGH = relay OFF (locked).

    Supports two modes:
      - **Pulse mode** (face auto-unlock): ``unlock(duration)`` energizes
        for N seconds then de-energizes.
      - **Persistent mode** (RF remote / manual): ``set_state(locked)``
        sets relay permanently until changed again.
    """

    def __init__(self, pin: int = 17):
        self.pin = pin
        self._gpio = _get_gpio()
        self._locked = True  # track persistent state

        if self._gpio is not None:
            try:
                self._gpio.setmode(self._gpio.BCM)
                self._gpio.setwarnings(False)
                # Active-low: HIGH = OFF = locked
                self._gpio.setup(self.pin, self._gpio.OUT, initial=self._gpio.HIGH)
                logger.info("Relay pin %d initialized HIGH (locked, active-low)", self.pin)
            except Exception as exc:
                logger.error("Failed to initialize relay on pin %d: %s", pin, exc)
                self._gpio = None
        else:
            logger.warning("RPi.GPIO unavailable — relay operations will be no-ops")

    # ── Pulse mode (face auto-unlock) ──────────────────────────────

    def unlock(self, duration: float = 3.0) -> bool:
        """Energize relay for *duration* seconds, then restore persistent state.

        Active-low: LOW = ON (unlocked).
        """
        if self._gpio is None:
            return False

        try:
            logger.info("Pulse unlock for %.1f s (pin %d LOW)", duration, self.pin)
            self._gpio.output(self.pin, self._gpio.LOW)
            time.sleep(duration)
            # Restore persistent state
            if self._locked:
                self._gpio.output(self.pin, self._gpio.HIGH)
                logger.info("Restored LOCKED state (HIGH)")
            else:
                self._gpio.output(self.pin, self._gpio.LOW)
                logger.info("Restored UNLOCKED state (LOW)")
            return True
        except Exception as exc:
            logger.error("Relay unlock failed on pin %d: %s", self.pin, exc)
            try:
                self._gpio.output(self.pin, self._gpio.HIGH)
            except Exception:
                pass
            return False

    # ── Persistent mode (RF remote / manual) ───────────────────────

    def set_state(self, locked: bool) -> None:
        """Set persistent relay state.

        Active-low: locked = HIGH (OFF), unlocked = LOW (ON).

        Args:
            locked: True = de-energize (locked), False = energize (unlocked).
        """
        self._locked = locked  # track state regardless of GPIO availability
        if self._gpio is None:
            return
        try:
            if locked:
                self._gpio.output(self.pin, self._gpio.HIGH)
                logger.info("Persistent state → LOCKED (pin %d HIGH)", self.pin)
            else:
                self._gpio.output(self.pin, self._gpio.LOW)
                logger.info("Persistent state → UNLOCKED (pin %d LOW)", self.pin)
        except Exception as exc:
            logger.error("Failed to set relay state: %s", exc)

    def lock_persistent(self) -> None:
        """Shortcut: set persistent state to LOCKED."""
        self.set_state(locked=True)

    def unlock_persistent(self) -> None:
        """Shortcut: set persistent state to UNLOCKED."""
        self.set_state(locked=False)

    def lock(self) -> None:
        """Immediately de-energize relay (locked / safe state).

        Equivalent to lock_persistent() but kept for backward compatibility.
        """
        self.lock_persistent()

    @property
    def is_locked(self) -> bool:
        """True if the persistent state is LOCKED, False if UNLOCKED."""
        return self._locked

    # ── Cleanup ────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Clean up GPIO resources (call on shutdown)."""
        if self._gpio is None:
            return
        try:
            # Always revert to locked (HIGH = OFF) on shutdown
            self._gpio.output(self.pin, self._gpio.HIGH)
            self._gpio.cleanup(self.pin)
            logger.info("GPIO cleanup for relay pin %d", self.pin)
        except Exception as exc:
            logger.warning("GPIO cleanup warning on pin %d: %s", self.pin, exc)
