"""
rf_receiver.py — 433MHz OOK remote receiver using GPIO edge-detection timing.

Uses RPi.GPIO edge detection (BOTH edges) to capture pulse-width timings
from a generic 433MHz OOK receiver module (FS1000A / MX-05V / XY-MK-5V).

When a valid burst of edges is detected (more than MIN_EDGES within
BURST_WINDOW_MS), fires the callback with 'LOCK' or 'UNLOCK'.

This approach works because 433MHz OOK receivers output a rapid series of
HIGH/LOW transitions when a button is pressed (Manchester/OOK data stream).
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
    433MHz OOK remote receiver via GPIO edge-detection with pulse-width timing.

    Records timestamps of every edge (BOTH rising and falling) and detects
    bursts characteristic of 433MHz OOK data transmissions.

    Attributes:
        lock_pin:   GPIO reading LOCK button receiver output.
        unlock_pin: GPIO reading UNLOCK button receiver output.
        is_configured: Always True (no code learning needed).
    """

    # Minimum edges in a burst to qualify as a valid RF transmission
    MIN_EDGES = 10
    # Time window (seconds) for a valid burst
    BURST_WINDOW_S = 0.10
    # Debounce interval (seconds) between successive triggers
    DEBOUNCE_S = 0.25

    def __init__(self, lock_pin: int = 22, unlock_pin: int = 23):
        """
        Args:
            lock_pin:   GPIO for LOCK receiver data pin.
            unlock_pin: GPIO for UNLOCK receiver data pin.
        """
        self.lock_pin = lock_pin
        self.unlock_pin = unlock_pin
        self._gpio = _get_gpio()
        self._callback: Optional[Callable[[str], None]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._is_configured = True  # always ready

        # Burst detection state per pin
        self._lock_edges: list[float] = []      # timestamps of last N edges
        self._unlock_edges: list[float] = []
        self._last_lock_ts = 0.0                 # last debounced trigger
        self._last_unlock_ts = 0.0
        self._lock_mutex = threading.Lock()
        self._unlock_mutex = threading.Lock()

        if self._gpio is not None:
            try:
                self._gpio.setmode(self._gpio.BCM)
                self._gpio.setwarnings(False)
                self._gpio.setup(self.lock_pin, self._gpio.IN, pull_up_down=self._gpio.PUD_DOWN)
                self._gpio.setup(self.unlock_pin, self._gpio.IN, pull_up_down=self._gpio.PUD_DOWN)
                logger.info(
                    "RF receiver: LOCK=GPIO%d, UNLOCK=GPIO%d (edge-detect)",
                    self.lock_pin, self.unlock_pin,
                )
            except Exception as exc:
                logger.error("Failed to init RF GPIOs: %s", exc)
                self._gpio = None
        else:
            logger.warning("RPi.GPIO unavailable — RF receiver disabled")

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    def set_callback(self, cb: Callable[[str], None]):
        """Set callback receiving 'LOCK' or 'UNLOCK' on button press."""
        self._callback = cb

    def start(self) -> bool:
        """Start the edge-detection background thread. Returns True on success."""
        if self._gpio is None:
            return False
        try:
            self._running = True
            self._thread = threading.Thread(target=self._run_edge_detect, daemon=True)
            self._thread.start()
            logger.info("RF edge-detect thread started (LOCK=GPIO%d, UNLOCK=GPIO%d)",
                        self.lock_pin, self.unlock_pin)
            return True
        except Exception as exc:
            logger.error("Failed to start RF receiver: %s", exc)
            self._running = False
            return False

    def stop(self):
        """Stop the edge-detection thread and clean up."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._gpio is not None:
            try:
                self._gpio.remove_event_detect(self.lock_pin)
            except Exception:
                pass
            try:
                self._gpio.remove_event_detect(self.unlock_pin)
            except Exception:
                pass
        logger.info("RF receiver stopped")

    # ── Internal: edge detection ──────────────────────────────────────

    def _on_lock_edge(self, channel):
        """Callback on any GPIO edge change on LOCK pin."""
        now = time.time()
        with self._lock_mutex:
            self._lock_edges.append(now)
            # Keep only edges within the burst window
            while self._lock_edges and (now - self._lock_edges[0]) > self.BURST_WINDOW_S:
                self._lock_edges.pop(0)
            edge_count = len(self._lock_edges)

        if edge_count >= self.MIN_EDGES and (now - self._last_lock_ts) > self.DEBOUNCE_S:
            self._last_lock_ts = now
            logger.info("RF: LOCK button pressed (%d edges in %.0fms)",
                        edge_count, self.BURST_WINDOW_S * 1000)
            with self._lock_mutex:
                self._lock_edges.clear()
            if self._callback:
                self._callback("LOCK")

    def _on_unlock_edge(self, channel):
        """Callback on any GPIO edge change on UNLOCK pin."""
        now = time.time()
        with self._unlock_mutex:
            self._unlock_edges.append(now)
            while self._unlock_edges and (now - self._unlock_edges[0]) > self.BURST_WINDOW_S:
                self._unlock_edges.pop(0)
            edge_count = len(self._unlock_edges)

        if edge_count >= self.MIN_EDGES and (now - self._last_unlock_ts) > self.DEBOUNCE_S:
            self._last_unlock_ts = now
            logger.info("RF: UNLOCK button pressed (%d edges in %.0fms)",
                        edge_count, self.BURST_WINDOW_S * 1000)
            with self._unlock_mutex:
                self._unlock_edges.clear()
            if self._callback:
                self._callback("UNLOCK")

    def _run_edge_detect(self):
        """Background thread: set up edge detection callbacks and keep alive."""
        try:
            self._gpio.add_event_detect(self.lock_pin, self._gpio.BOTH,
                                        callback=self._on_lock_edge)
            self._gpio.add_event_detect(self.unlock_pin, self._gpio.BOTH,
                                        callback=self._on_unlock_edge)
        except Exception as exc:
            logger.error("Failed to add edge detection: %s", exc)
            self._running = False
            return

        logger.info("RF edge detection active on GPIO%d, GPIO%d",
                    self.lock_pin, self.unlock_pin)

        # Keep thread alive; callbacks fire from a separate RPi.GPIO thread
        while self._running:
            try:
                time.sleep(0.5)
            except Exception:
                break
