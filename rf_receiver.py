"""
rf_receiver.py — 433MHz remote receiver using wait_for_edge with pulse-width timing.

Uses RPi.GPIO.wait_for_edge() in a background thread to detect any signal
change (BOTH edges), then records timestamps to identify valid RF bursts.

Handles two detection modes:
1. Edge burst (OOK data stream): ≥2 edges within 150ms
2. Falling hold (VT-type): pin stays LOW for >50ms after first falling edge
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
    433MHz remote receiver using wait_for_edge with pulse-width timing.

    Attributes:
        lock_pin:   GPIO reading LOCK button receiver output.
        unlock_pin: GPIO reading UNLOCK button receiver output.
        is_configured: Always True (no code learning needed).
    """

    MIN_EDGES = 2
    BURST_WINDOW_S = 0.15
    DEBOUNCE_S = 0.30
    EDGE_TIMEOUT_MS = 500  # wait_for_edge timeout in ms

    def __init__(self, lock_pin: int = 22, unlock_pin: int = 23):
        self.lock_pin = lock_pin
        self.unlock_pin = unlock_pin
        self._gpio = _get_gpio()
        self._callback: Optional[Callable[[str], None]] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._is_configured = True

        # Edge timestamp tracking per pin
        self._lock_edges: list[float] = []
        self._unlock_edges: list[float] = []
        self._last_lock_ts = 0.0
        self._last_unlock_ts = 0.0
        self._lock_mutex = threading.Lock()
        self._unlock_mutex = threading.Lock()

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    def set_callback(self, cb: Callable[[str], None]):
        self._callback = cb

    def start(self) -> bool:
        if self._gpio is None:
            return False
        try:
            self._gpio.setmode(self._gpio.BCM)
            self._gpio.setwarnings(False)
            self._gpio.setup(self.lock_pin, self._gpio.IN, pull_up_down=self._gpio.PUD_DOWN)
            self._gpio.setup(self.unlock_pin, self._gpio.IN, pull_up_down=self._gpio.PUD_DOWN)

            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
            logger.info("RF receiver started (LOCK=GPIO%d, UNLOCK=GPIO%d)",
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
        if self._gpio is not None:
            self._gpio.cleanup(self.lock_pin)
            self._gpio.cleanup(self.unlock_pin)
        logger.info("RF receiver stopped")

    # ── Internal: polling loop ───────────────────────────────────────

    def _poll_loop(self):
        """
        Background thread: poll both pins using wait_for_edge.

        Detects edges on either pin, records timestamps, and checks
        for valid RF bursts using burst-count or falling-hold modes.
        """
        while self._running:
            try:
                # Block until edge on either pin, with timeout for _running check
                channel = self._gpio.wait_for_edge(
                    self.lock_pin, self._gpio.BOTH,
                    timeout=self.EDGE_TIMEOUT_MS
                )
                if channel is None:
                    continue  # timeout, re-check _running

                now = time.time()

                if channel == self.lock_pin:
                    self._record_edge(now, self._lock_edges, self._lock_mutex)
                    self._last_lock_ts = self._check_trigger(
                        "LOCK", self._lock_edges, self._lock_mutex,
                        self._last_lock_ts, self.lock_pin
                    )
                elif channel == self.unlock_pin:
                    self._record_edge(now, self._unlock_edges, self._unlock_mutex)
                    self._last_unlock_ts = self._check_trigger(
                        "UNLOCK", self._unlock_edges, self._unlock_mutex,
                        self._last_unlock_ts, self.unlock_pin
                    )

                # Poll the other pin non-blocking too
                if self._running:
                    try:
                        channel2 = self._gpio.wait_for_edge(
                            self.unlock_pin, self._gpio.BOTH,
                            timeout=1
                        )
                        if channel2 == self.unlock_pin:
                            now2 = time.time()
                            self._record_edge(now2, self._unlock_edges, self._unlock_mutex)
                            self._last_unlock_ts = self._check_trigger(
                                "UNLOCK", self._unlock_edges, self._unlock_mutex,
                                self._last_unlock_ts, self.unlock_pin
                            )
                    except Exception:
                        pass

            except Exception as exc:
                logger.error("RF poll loop error: %s", exc)
                time.sleep(0.1)

    def _record_edge(self, now: float, edges: list, mutex: threading.Lock):
        with mutex:
            edges.append(now)
            while edges and (now - edges[0]) > self.BURST_WINDOW_S:
                edges.pop(0)

    def _check_trigger(self, pin_name: str, edges: list, mutex: threading.Lock,
                       last_ts: float, gpio_pin: int) -> float:
        now = time.time()
        if (now - last_ts) < self.DEBOUNCE_S:
            return last_ts

        with mutex:
            edge_count = len(edges)

        # Mode 1: edge burst (OOK data stream)
        if edge_count >= self.MIN_EDGES:
            logger.info("RF: %s pressed (%d edges in %.0fms)",
                        pin_name, edge_count, self.BURST_WINDOW_S * 1000)
            with mutex:
                edges.clear()
            if self._callback:
                self._callback(pin_name)
            return now

        # Mode 2: falling hold (pin stays LOW >50ms)
        if edge_count >= 1:
            try:
                pin_val = bool(self._gpio.input(gpio_pin))
            except Exception:
                pin_val = True
            if not pin_val:
                hold_time = now - edges[0]
                if hold_time > 0.05:
                    logger.info("RF: %s pressed (VT hold %.0fms, %d edges)",
                                pin_name, hold_time * 1000, edge_count)
                    with mutex:
                        edges.clear()
                    if self._callback:
                        self._callback(pin_name)
                    return now

        return last_ts
