"""
rf_receiver.py — 433MHz remote receiver via high-speed GPIO polling thread.

Uses a background thread polling both GPIO pins at 2ms intervals (500Hz)
to detect voltage transitions. No RPi.GPIO edge detection needed.

Two detection modes:
1. Edge burst (OOK data): ≥4 edges within 150ms
2. Falling hold (VT-type): pin stays LOW >50ms
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
    """
    433MHz remote receiver via high-speed GPIO polling thread.

    Polls GPIO pins at 500Hz to detect voltage transitions caused by
    433MHz OOK receiver modules.

    Attributes:
        lock_pin:   GPIO reading LOCK button receiver output.
        unlock_pin: GPIO reading UNLOCK button receiver output.
        is_configured: Always True (no code learning needed).
    """

    POLL_INTERVAL = 0.002   # 2ms = 500Hz
    MIN_EDGES = 4           # minimum transitions for a valid OOK burst
    BURST_WINDOW_S = 0.15   # time window for counting edges
    HOLD_THRESHOLD_S = 0.05 # LOW hold time for VT mode
    DEBOUNCE_S = 0.30       # min time between successive triggers

    def __init__(self, lock_pin: int = 22, unlock_pin: int = 23):
        self.lock_pin = lock_pin
        self.unlock_pin = unlock_pin
        self._gpio = _get_gpio()
        self._callback: Optional[Callable[[str], None]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._is_configured = True

        # Previous poll values for transition detection
        self._prev_lock = False
        self._prev_unlock = False

        # Edge timestamps for OOK burst detection
        self._lock_edges: list[float] = []
        self._unlock_edges: list[float] = []

        # Last trigger timestamps (debounce)
        self._last_lock_ts = 0.0
        self._last_unlock_ts = 0.0

        if self._gpio is not None:
            try:
                self._gpio.setmode(self._gpio.BCM)
                self._gpio.setwarnings(False)
                self._gpio.setup(self.lock_pin, self._gpio.IN, pull_up_down=self._gpio.PUD_DOWN)
                self._gpio.setup(self.unlock_pin, self._gpio.IN, pull_up_down=self._gpio.PUD_DOWN)
                logger.info("RF receiver: LOCK=GPIO%d, UNLOCK=GPIO%d (500Hz poll)",
                            self.lock_pin, self.unlock_pin)
            except Exception as exc:
                logger.error("Failed to init RF GPIOs: %s", exc)
                self._gpio = None
        else:
            logger.warning("RPi.GPIO unavailable — RF receiver disabled")

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    def set_callback(self, cb: Callable[[str], None]):
        self._callback = cb

    def start(self) -> bool:
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
            logger.error("Failed to start RF receiver: %s", exc)
            self._running = False
            return False

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("RF receiver stopped")

    # ── Internal: polling loop ─────────────────────────────────────

    def _poll_loop(self):
        """Background thread: poll GPIOs at 500Hz, detect transitions."""
        # Read initial state to avoid false trigger
        try:
            self._prev_lock = bool(self._gpio.input(self.lock_pin))
            self._prev_unlock = bool(self._gpio.input(self.unlock_pin))
        except Exception:
            self._prev_lock = False
            self._prev_unlock = False

        while self._running:
            now = time.time()
            try:
                curr_lock = bool(self._gpio.input(self.lock_pin))
                curr_unlock = bool(self._gpio.input(self.unlock_pin))
            except Exception:
                time.sleep(self.POLL_INTERVAL)
                continue

            # ── LOCK ─────────────────────────────────────────────
            if curr_lock != self._prev_lock:
                self._lock_edges.append(now)
                while self._lock_edges and (now - self._lock_edges[0]) > self.BURST_WINDOW_S:
                    self._lock_edges.pop(0)
                edges = len(self._lock_edges)
                if (now - self._last_lock_ts) > self.DEBOUNCE_S:
                    triggered = False
                    # Mode 1: edge burst
                    if edges >= self.MIN_EDGES:
                        triggered = True
                    # Mode 2: falling hold
                    elif edges >= 1 and not curr_lock:
                        hold = now - self._lock_edges[0]
                        if hold > self.HOLD_THRESHOLD_S:
                            triggered = True
                    if triggered:
                        self._last_lock_ts = now
                        logger.info("RF: LOCK (edges=%d, pin=%d)", edges, int(curr_lock))
                        self._lock_edges.clear()
                        if self._callback:
                            self._callback("LOCK")

            # ── UNLOCK ───────────────────────────────────────────
            if curr_unlock != self._prev_unlock:
                self._unlock_edges.append(now)
                while self._unlock_edges and (now - self._unlock_edges[0]) > self.BURST_WINDOW_S:
                    self._unlock_edges.pop(0)
                edges = len(self._unlock_edges)
                if (now - self._last_unlock_ts) > self.DEBOUNCE_S:
                    triggered = False
                    # Mode 1: edge burst
                    if edges >= self.MIN_EDGES:
                        triggered = True
                    # Mode 2: falling hold
                    elif edges >= 1 and not curr_unlock:
                        hold = now - self._unlock_edges[0]
                        if hold > self.HOLD_THRESHOLD_S:
                            triggered = True
                    if triggered:
                        self._last_unlock_ts = now
                        logger.info("RF: UNLOCK (edges=%d, pin=%d)", edges, int(curr_unlock))
                        self._unlock_edges.clear()
                        if self._callback:
                            self._callback("UNLOCK")

            self._prev_lock = curr_lock
            self._prev_unlock = curr_unlock
            time.sleep(self.POLL_INTERVAL)
