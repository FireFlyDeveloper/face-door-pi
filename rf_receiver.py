"""
rf_receiver.py — Two-button 433MHz remote via GPIO polling thread.

Uses a background polling thread instead of RPi.GPIO edge detection,
which can be unreliable with noisy 433MHz OOK data stream signals.

GPIO mapping:
  lock_pin   (default GPIO22) → LOCK action
  unlock_pin (default GPIO23) → UNLOCK action

Pull-down resistors ensure stable LOW when no button is pressed.
Debounce of ~150ms prevents false triggers from RF noise or burst edges.
"""

from __future__ import annotations

import logging
import threading
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
    """Two-button 433MHz remote via GPIO polling thread.

    Supports per-pin polarity: 'rising' (LOW→HIGH) or 'falling' (HIGH→LOW).

    Attributes:
        lock_pin:   GPIO reading LOCK button receiver output.
        unlock_pin: GPIO reading UNLOCK button receiver output.
        is_configured: Always True (no code learning needed).
    """

    POLL_INTERVAL = 0.02   # 50 Hz poll rate
    DEBOUNCE_S = 0.15      # minimum interval between triggers

    def __init__(self, lock_pin: int = 22, unlock_pin: int = 23,
                 lock_polarity: str = 'rising', unlock_polarity: str = 'falling'):
        """
        Args:
            lock_pin: GPIO for LOCK receiver data pin.
            unlock_pin: GPIO for UNLOCK receiver data pin.
            lock_polarity: 'rising' (LOW→HIGH) or 'falling' (HIGH→LOW).
            unlock_polarity: 'rising' or 'falling'.
        """
        self.lock_pin = lock_pin
        self.unlock_pin = unlock_pin
        self._lock_polarity = lock_polarity
        self._unlock_polarity = unlock_polarity
        self._gpio = _get_gpio()
        self._callback: Optional[Callable[[str], None]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
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
                    "RF receiver: LOCK=GPIO%d, UNLOCK=GPIO%d (pull-down, poll)",
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
        """Start the polling background thread. Returns True on success."""
        if self._gpio is None:
            return False
        try:
            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
            logger.info("RF poll thread started (LOCK=GPIO%d, UNLOCK=GPIO%d)",
                        self.lock_pin, self.unlock_pin)
            return True
        except Exception as exc:
            logger.error("Failed to start RF poll thread: %s", exc)
            self._running = False
            return False

    def stop(self):
        """Stop the polling thread and clean up."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._gpio is not None:
            try:
                self._gpio.cleanup(self.lock_pin)
                self._gpio.cleanup(self.unlock_pin)
            except Exception:
                pass
        logger.info("RF receiver stopped")

    # ── Internal: polling loop ─────────────────────────────────────

    def _poll_loop(self):
        """Background thread: poll both GPIOs at POLL_INTERVAL rate.

        Uses transition detection with per-pin polarity:
          'rising'  → fires on LOW→HIGH
          'falling' → fires on HIGH→LOW
        """
        prev_lock = False
        prev_unlock = False

        # Quick initial read to avoid false trigger on first poll
        try:
            prev_lock = bool(self._gpio.input(self.lock_pin))
            prev_unlock = bool(self._gpio.input(self.unlock_pin))
        except Exception:
            pass

        while self._running:
            now = time.time()
            curr_lock = bool(self._gpio.input(self.lock_pin))
            curr_unlock = bool(self._gpio.input(self.unlock_pin))

            # ── LOCK ─────────────────────────────────────────────
            if self._lock_polarity == 'falling':
                lock_active = prev_lock and not curr_lock   # HIGH→LOW
            else:
                lock_active = curr_lock and not prev_lock   # LOW→HIGH
            if lock_active and (now - self._last_lock_ts) > self.DEBOUNCE_S:
                self._last_lock_ts = now
                logger.info("RF: LOCK button pressed")
                if self._callback:
                    self._callback("LOCK")

            # ── UNLOCK ───────────────────────────────────────────
            if self._unlock_polarity == 'falling':
                unlock_active = prev_unlock and not curr_unlock   # HIGH→LOW
            else:
                unlock_active = curr_unlock and not prev_unlock   # LOW→HIGH
            if unlock_active and (now - self._last_unlock_ts) > self.DEBOUNCE_S:
                self._last_unlock_ts = now
                logger.info("RF: UNLOCK button pressed")
                if self._callback:
                    self._callback("UNLOCK")

            prev_lock = curr_lock
            prev_unlock = curr_unlock
            time.sleep(self.POLL_INTERVAL)
