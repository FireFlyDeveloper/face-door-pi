"""
relay_controller.py — 2-relay H-bridge for DC motor polarity reversal.

Controls a car door actuator (brushed DC motor) via two relays arranged as
an H-bridge. Active-low relay modules: LOW=ON(energized), HIGH=OFF.

Truth table (active-low relays):
  R1(GPIO17)  R2(GPIO27)  Motor
  HIGH        HIGH        STOP/Brake (idle)
  LOW         HIGH        Clockwise  (UNLOCK)
  HIGH        LOW         Counter-CW (LOCK)

Always returns to STOP after any pulse to prevent motor burn-out
(no limit switch sensor on the actuator).
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


# ── Active-low relay constants ─────────────────────────────────────────
_RELAY_OFF = True   # GPIO HIGH → relay de-energized (OFF)
_RELAY_ON = False   # GPIO LOW  → relay energized (ON)


class RelayController:
    """2-relay H-bridge controller for DC motor door actuator.

    Three motor states: STOP (idle/brake), UNLOCK (CW), LOCK (CCW).
    Always auto-returns to STOP after a timed pulse.
    """

    def __init__(self, pin1: int = 17, pin2: int = 27):
        """Initialize two relay GPIOs.

        Args:
            pin1: BCM pin for Relay 1 (default 17).
            pin2: BCM pin for Relay 2 (default 27).
        """
        self.pin1 = pin1
        self.pin2 = pin2
        self._gpio = _get_gpio()

        if self._gpio is not None:
            try:
                self._gpio.setmode(self._gpio.BCM)
                self._gpio.setwarnings(False)
                self._gpio.setup(self.pin1, self._gpio.OUT, initial=self._gpio.HIGH)
                self._gpio.setup(self.pin2, self._gpio.OUT, initial=self._gpio.HIGH)
                logger.info(
                    "H-bridge relays: R1=GPIO%d(HIGH=OFF), R2=GPIO%d(HIGH=OFF) — STOP",
                    self.pin1, self.pin2,
                )
            except Exception as exc:
                logger.error("Failed to init H-bridge relays: %s", exc)
                self._gpio = None
        else:
            logger.warning("RPi.GPIO unavailable — relay operations are no-ops")

    # ── Internal: set both relays ───────────────────────────────────

    def _set(self, r1_state: bool, r2_state: bool):
        """Set both relays. True=OFF(HIGH), False=ON(LOW)."""
        if self._gpio is None:
            return
        try:
            self._gpio.output(self.pin1, self._gpio.HIGH if r1_state else self._gpio.LOW)
            self._gpio.output(self.pin2, self._gpio.HIGH if r2_state else self._gpio.LOW)
        except Exception as exc:
            logger.error("H-bridge _set error: %s", exc)

    # ── STOP ────────────────────────────────────────────────────────

    def stop(self) -> None:
        """Immediately brake the motor (both relays OFF)."""
        logger.debug("H-bridge: STOP (R1=OFF R2=OFF)")
        self._set(_RELAY_OFF, _RELAY_OFF)

    # ── UNLOCK pulse (Clockwise) ────────────────────────────────────

    def unlock_cw(self, duration: float = 1.0) -> bool:
        """Run motor clockwise (UNLOCK) for *duration*, then STOP.

        Args:
            duration: Seconds to run motor (default 1.0).

        Returns:
            True if successful.
        """
        if self._gpio is None:
            return False
        try:
            logger.info("H-bridge: UNLOCK (CW) for %.1f s", duration)
            self._set(_RELAY_ON, _RELAY_OFF)  # R1=ON, R2=OFF → CW
            time.sleep(duration)
            self.stop()
            return True
        except Exception as exc:
            logger.error("H-bridge unlock_cw error: %s", exc)
            try:
                self.stop()
            except Exception:
                pass
            return False

    # ── LOCK pulse (Counter-clockwise) ──────────────────────────────

    def lock_ccw(self, duration: float = 1.0) -> bool:
        """Run motor counter-clockwise (LOCK) for *duration*, then STOP.

        Args:
            duration: Seconds to run motor (default 1.0).

        Returns:
            True if successful.
        """
        if self._gpio is None:
            return False
        try:
            logger.info("H-bridge: LOCK (CCW) for %.1f s", duration)
            self._set(_RELAY_OFF, _RELAY_ON)  # R1=OFF, R2=ON → CCW
            time.sleep(duration)
            self.stop()
            return True
        except Exception as exc:
            logger.error("H-bridge lock_ccw error: %s", exc)
            try:
                self.stop()
            except Exception:
                pass
            return False

    # ── Cleanup ─────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """Stop motor and release GPIO resources."""
        if self._gpio is None:
            return
        try:
            self.stop()
            self._gpio.cleanup(self.pin1)
            self._gpio.cleanup(self.pin2)
            logger.info("H-bridge GPIO cleanup done")
        except Exception as exc:
            logger.warning("H-bridge cleanup warning: %s", exc)
