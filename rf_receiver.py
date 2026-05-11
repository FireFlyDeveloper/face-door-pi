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
    bursts characteristic of 433MHz OOK data transmissions, OR a simple
    voltage drop (single pulse) from VT-type receiver modules.

    Attributes:
        lock_pin:   GPIO reading LOCK button receiver output.
        unlock_pin: GPIO reading UNLOCK button receiver output.
        is_configured: Always True (no code learning needed).
    """

    # Minimum edges in a burst to qualify as a valid RF transmission
    # Set low (2) to catch a single voltage-drop pulse (2 edges = HIGH→LOW→HIGH)
    MIN_EDGES = 2
    # Time window (seconds) for a valid burst
    BURST_WINDOW_S = 0.15
    # Debounce interval (seconds) between successive triggers
    DEBOUNCE_S = 0.30

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

    def _check_and_trigger(self, pin_name: str, edges: list, mutex: threading.Lock,
                           last_ts: float, gpio_pin: int) -> float:
        """
        Check if a valid signal was received and trigger callback.

        Two detection modes:
        1. Edge burst: ≥MIN_EDGES edges within BURST_WINDOW_S (OOK data stream)
        2. Falling hold: current pin is LOW for >50ms after first edge (VT pulse)

        Returns updated last_ts.
        """
        now = time.time()
        with mutex:
            while edges and (now - edges[0]) > self.BURST_WINDOW_S:
                edges.pop(0)
            edge_count = len(edges)

        # Mode 1: edge burst detected
        if edge_count >= self.MIN_EDGES and (now - last_ts) > self.DEBOUNCE_S:
            logger.info("RF: %s pressed (%d edges in %.0fms)",
                        pin_name, edge_count, self.BURST_WINDOW_S * 1000)
            with mutex:
                edges.clear()
            if self._callback:
                self._callback(pin_name)
            return now

        # Mode 2: falling-edge hold (pin is LOW, has been for a while)
        if edge_count >= 1 and (now - last_ts) > self.DEBOUNCE_S:
            try:
                pin_val = bool(self._gpio.input(gpio_pin))
            except Exception:
                pin_val = True  # assume HIGH on error, don't trigger
            if not pin_val:
                # Pin is LOW and has been for >=1 edge cycle + debounce check
                hold_time = now - edges[0]
                if hold_time > 0.05:  # held LOW for >50ms
                    logger.info("RF: %s pressed (VT hold %.0fms, %d edges)",
                                pin_name, hold_time * 1000, edge_count)
                    with mutex:
                        edges.clear()
                    if self._callback:
                        self._callback(pin_name)
                    return now

        return last_ts

    def _on_lock_edge(self, channel):
        """Callback on any GPIO edge change on LOCK pin."""
        with self._lock_mutex:
            self._lock_edges.append(time.time())
        self._last_lock_ts = self._check_and_trigger(
            "LOCK", self._lock_edges, self._lock_mutex,
            self._last_lock_ts, self.lock_pin
        )

    def _on_unlock_edge(self, channel):
        """Callback on any GPIO edge change on UNLOCK pin."""
        with self._unlock_mutex:
            self._unlock_edges.append(time.time())
        self._last_unlock_ts = self._check_and_trigger(
            "UNLOCK", self._unlock_edges, self._unlock_mutex,
            self._last_unlock_ts, self.unlock_pin
        )

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
