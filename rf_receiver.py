"""
rf_receiver.py — Two-button 433MHz remote via direct GPIO edge detection.

Two separate receiver modules (or a dual-channel receiver) each output to
a dedicated GPIO pin. Rising-edge detection triggers the corresponding
callback — no rpi-rf code decoding needed.

GPIO mapping:
  lock_pin   (default GPIO22) → LOCK action
  unlock_pin (default GPIO23) → UNLOCK action

Pull-down resistors ensure stable LOW when no button is pressed.
Debounce of ~100ms prevents false triggers from RF noise.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Callable

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


class RFReceiver:
    """Two-button 433MHz remote via direct GPIO edge detection.

    Attributes:
        lock_pin:   GPIO reading LOCK button receiver output.
        unlock_pin: GPIO reading UNLOCK button receiver output.
        is_configured: Always True (no code learning needed).
    """

    DEBOUNCE_MS = 150  # minimum interval between triggers

    def __init__(self, lock_pin: int = 22, unlock_pin: int = 23):
        self.lock_pin = lock_pin
        self.unlock_pin = unlock_pin
        self._gpio = _get_gpio()
        self._callback: Optional[Callable[[str], None]] = None
        self._running = False
        self._last_lock_ts = 0.0
        self._last_unlock_ts = 0.0
        self.is_configured = True  # always ready

        if self._gpio is not None:
            try:
                self._gpio.setmode(self._gpio.BCM)
                self._gpio.setwarnings(False)
                self._gpio.setup(self.lock_pin, self._gpio.IN, pull_up_down=self._gpio.PUD_DOWN)
                self._gpio.setup(self.unlock_pin, self._gpio.IN, pull_up_down=self._gpio.PUD_DOWN)
                logger.info(
                    "RF receiver: LOCK=GPIO%d, UNLOCK=GPIO%d (pull-down)",
                    self.lock_pin, self.unlock_pin,
                )
            except Exception as exc:
                logger.error("Failed to init RF GPIOs: %s", exc)
                self._gpio = None
        else:
            logger.warning("RPi.GPIO unavailable — RF receiver disabled")

    def set_callback(self, cb: Callable[[str], None]):
        """Set callback receiving 'LOCK' or 'UNLOCK' on button press."""
        self._callback = cb

    def start(self) -> bool:
        """Attach rising-edge interrupts for both pins. Returns True on success."""
        if self._gpio is None:
            return False
        try:
            self._running = True
            self._gpio.add_event_detect(
                self.lock_pin, self._gpio.RISING,
                callback=self._lock_handler, bouncetime=self.DEBOUNCE_MS,
            )
            self._gpio.add_event_detect(
                self.unlock_pin, self._gpio.RISING,
                callback=self._unlock_handler, bouncetime=self.DEBOUNCE_MS,
            )
            logger.info("RF edge detection started (LOCK=GPIO%d, UNLOCK=GPIO%d)",
                        self.lock_pin, self.unlock_pin)
            return True
        except Exception as exc:
            logger.error("Failed to start RF edge detection: %s", exc)
            self._running = False
            return False

    def stop(self):
        """Remove event detection and clean up."""
        self._running = False
        if self._gpio is None:
            return
        try:
            self._gpio.remove_event_detect(self.lock_pin)
        except Exception:
            pass
        try:
            self._gpio.remove_event_detect(self.unlock_pin)
        except Exception:
            pass
        try:
            self._gpio.cleanup(self.lock_pin)
            self._gpio.cleanup(self.unlock_pin)
        except Exception:
            pass
        logger.info("RF receiver stopped")

    # ── Internal handlers ──────────────────────────────────────────

    def _lock_handler(self, channel: int):
        now = time.time()
        if (now - self._last_lock_ts) * 1000 < self.DEBOUNCE_MS:
            return
        self._last_lock_ts = now
        logger.info("RF: LOCK button pressed")
        if self._callback:
            self._callback("LOCK")

    def _unlock_handler(self, channel: int):
        now = time.time()
        if (now - self._last_unlock_ts) * 1000 < self.DEBOUNCE_MS:
            return
        self._last_unlock_ts = now
        logger.info("RF: UNLOCK button pressed")
        if self._callback:
            self._callback("UNLOCK")
