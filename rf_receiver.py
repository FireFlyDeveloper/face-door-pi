"""
rf_receiver.py — 433MHz RF remote receiver for manual lock/unlock.

Monitors a 433MHz receiver module on a configurable GPIO pin and decodes
common 433MHz protocols (PT2262, SC5262, etc.) via the rpi-rf library.

Maps received codes to LOCK / UNLOCK actions. Supports a learn mode
to capture remote button codes at runtime.
"""

from __future__ import annotations

import json
import os
import threading
import time
import logging
from typing import Optional, Callable

logger = logging.getLogger(__name__)

# ── Lazy rpi-rf import ───────────────────────────────────────────────
_RF_AVAILABLE = False
RFDevice = None

try:
    from rpi_rf import RFDevice as _RFDevice

    _RF_AVAILABLE = True
    RFDevice = _RFDevice
except ImportError:
    logger.warning(
        "rpi-rf not installed. 433MHz receiver disabled. "
        "Install: pip3 install rpi-rf"
    )

# ── Default config path ─────────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "rf_codes.json")


class RFReceiver:
    """
    Listens to a 433MHz receiver on *pin* (BCM).
    Calls *callback* with 'LOCK' or 'UNLOCK' when a known code arrives.

    Known codes are stored in rf_codes.json:
        {"lock": 1234567, "unlock": 7654321, "pulselen": 350}
    """

    def __init__(
        self,
        pin: int = 22,
        config_path: str = CONFIG_PATH,
    ):
        self.pin = pin
        self.config_path = config_path
        self._callback: Optional[Callable[[str], None]] = None  # arg: 'LOCK' | 'UNLOCK'
        self._learn_callback: Optional[Callable[[int, int], None]] = None  # arg: code, pulselen
        self._device: Optional[RFDevice] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # Load known codes
        self._lock_code: Optional[int] = None
        self._unlock_code: Optional[int] = None
        self._pulselen: Optional[int] = None
        self._load_codes()

    # ── Public API ───────────────────────────────────────────────────

    def set_callback(self, cb: Callable[[str], None]):
        """Receive 'LOCK' or 'UNLOCK' when a known RF code arrives."""
        self._callback = cb

    def set_learn_callback(self, cb: Callable[[int, int], None]):
        """
        Receive (code, pulselen) for *any* RF burst.
        Used during learn mode to discover remote button codes.
        """
        self._learn_callback = cb

    def start(self) -> bool:
        """Start the RF receiver thread. Returns True if started."""
        if not _RF_AVAILABLE or RFDevice is None:
            logger.warning("rpi-rf unavailable — RF receiver not started")
            return False

        try:
            self._device = RFDevice(self.pin)
            self._device.enable_rx()
            self._running = True
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
            logger.info("RF receiver started on GPIO%d", self.pin)
            return True
        except Exception as exc:
            logger.error("Failed to start RF receiver on GPIO%d: %s", self.pin, exc)
            return False

    def stop(self):
        """Stop the RF receiver and clean up."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._device:
            try:
                self._device.cleanup()
            except Exception:
                pass
        logger.info("RF receiver stopped")

    def save_code(self, code: int, pulselen: int, action: str):
        """
        Save a learned code to rf_codes.json.

        Args:
            code: The RF code value.
            pulselen: Pulse length in microseconds.
            action: 'lock' or 'unlock'.
        """
        codes = self._load_raw()
        codes[action] = code
        codes["pulselen"] = pulselen
        self._save_raw(codes)
        self._lock_code = codes.get("lock")
        self._unlock_code = codes.get("unlock")
        self._pulselen = codes.get("pulselen")
        logger.info("Saved RF code %d → %s", code, action)

    @property
    def is_configured(self) -> bool:
        """True if both lock and unlock codes are configured."""
        return self._lock_code is not None and self._unlock_code is not None

    # ── Internal ─────────────────────────────────────────────────────

    def _load_codes(self):
        codes = self._load_raw()
        self._lock_code = codes.get("lock")
        self._unlock_code = codes.get("unlock")
        self._pulselen = codes.get("pulselen")

    def _load_raw(self) -> dict:
        try:
            with open(self.config_path) as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_raw(self, codes: dict):
        os.makedirs(os.path.dirname(self.config_path) or ".", exist_ok=True)
        with open(self.config_path, "w") as f:
            json.dump(codes, f, indent=2)

    def _poll_loop(self):
        """Background thread: poll RF device for new codes."""
        device = self._device
        last_ts = 0

        while self._running:
            try:
                ts = device.rx_code_timestamp
                if ts != 0 and ts != last_ts:
                    last_ts = ts
                    code = device.rx_code
                    pulselen = device.rx_pulselen
                    self._handle_code(code, pulselen)
                time.sleep(0.02)  # 50 Hz poll
            except Exception as exc:
                logger.debug("RF poll error: %s", exc)
                time.sleep(0.1)

    def _handle_code(self, code: int, pulselen: int):
        """Route a received code to the appropriate handler."""
        # Always notify learn callback if set (learn mode)
        if self._learn_callback:
            self._learn_callback(code, pulselen)
            return  # don't map during learn mode

        # Map known codes
        if self._lock_code is not None and code == self._lock_code:
            logger.info("RF: LOCK code received")
            if self._callback:
                self._callback("LOCK")
        elif self._unlock_code is not None and code == self._unlock_code:
            logger.info("RF: UNLOCK code received")
            if self._callback:
                self._callback("UNLOCK")
        else:
            logger.info("RF: unknown code %d (pulselen=%d)", code, pulselen)
