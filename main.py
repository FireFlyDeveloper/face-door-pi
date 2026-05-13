#!/usr/bin/env python3
"""
Main entry point for the face recognition door system.
Revised pipeline: SCANNING -> COMPARE (> anti-spoof + ArcFace) -> GRANTED/REJECTED
Single-frame pipeline, no multi-frame collection needed.
"""

import sys
import os
import time
import json
import signal
import base64
import traceback
import io
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Optional

# Ensure project directory is on path for sibling module imports
PROJECT_DIR = '/home/admin/face-door-system'
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ── Imports of sibling modules ──────────────────────────────────────────
try:
    from camera_controller import CameraController
    from relay_controller import RelayController
    from buzzer_controller import BuzzerController
    from face_storage import FaceStorage
    from face_recognizer import FaceRecognizer
    from rf_receiver import RFReceiver
    from anti_spoof import AntiSpoofDetector, SCORE_THRESHOLD as LIVE_THRESHOLD
except ImportError as e:
    print(f"[Main] FATAL: Could not import sibling modules: {e}")
    print("[Main] Make sure all modules exist in", PROJECT_DIR)
    sys.exit(1)

from bluetooth_server import BluetoothServer
from logger import ActivityLogger
from metrics import MetricsLogger


# ── Constants ───────────────────────────────────────────────────────────
FRAME_RATE = 15.0
FRAME_INTERVAL = 1.0 / FRAME_RATE
UNLOCK_DURATION = 0.5
MATCH_THRESHOLD = 0.6        # cosine similarity threshold


# ── State Machine ──────────────────────────────────────────────────────
class State:
    INIT = "INIT"
    SCANNING = "SCANNING"
    COMPARE = "COMPARE"
    GRANTED = "GRANTED"
    REJECTED = "REJECTED"


# CV2 display window name
CV2_WINDOW = "Face Door System"

# ── CLI args ─────────────────────────────────────────────────────────
HEADLESS = "--headless" in sys.argv or "-H" in sys.argv
PREVIEW = "--preview" in sys.argv or "-p" in sys.argv


class FaceDoorSystem:
    """Main door system orchestrator with simplified state machine."""

    def __init__(self):
        self.state = State.INIT
        self._running = True
        self._latest_frame = None
        self._matched_id = None
        self._last_distance = 0.0
        self._frame_count = 0
        self._fps_timer = time.time()
        self._fps_counter = 0

        self.camera = None
        self.relay = None
        self.buzzer = None
        self.face_storage = None
        self.face_recognizer = None
        self.anti_spoof = None
        self.bt_server = None
        self.logger = None
        self.rf_receiver = None
        self.metrics = None

        # Persistent lock state (software tracking for UI)
        self._lock_state = "locked"  # "locked" | "unlocked"

        # Pending multi-step registration encodings
        self._pending_encodings: Dict[str, List[np.ndarray]] = {}

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    # ── Signal Handler ──────────────────────────────────────────────
    def _signal_handler(self, signum, frame):
        print(f"\n[Main] Signal {signum} received, shutting down...")
        self._running = False

    # ── Initialization ───────────────────────────────────────────────
    def _init_controllers(self):
        """Create and start all controllers. Called once in INIT state."""
        print("[Main] Initializing controllers...")

        # Preview window — only if a display is available
        if PREVIEW or (not HEADLESS and os.environ.get("DISPLAY", "")):
            try:
                test_img = np.zeros((10, 10, 3), dtype=np.uint8)
                cv2.imshow(CV2_WINDOW + "_test", test_img)
                cv2.waitKey(1)
                cv2.destroyWindow(CV2_WINDOW + "_test")
                self._show_preview_enabled = True
                print("[Main] Preview window enabled (press 'q' to quit)")
            except Exception:
                self._show_preview_enabled = False
                print("[Main] Preview not available — install python3-opencv (system pkg)")
        else:
            self._show_preview_enabled = False

        self.camera = CameraController()
        if not self.camera.start():
            print("[Main] FATAL: Camera failed to start")
            return False

        self.relay = RelayController(pin1=17, pin2=27)
        self.buzzer = BuzzerController()
        self.face_storage = FaceStorage()
        self.face_recognizer = FaceRecognizer(threshold=MATCH_THRESHOLD)
        self.logger = ActivityLogger()

        # Single-frame MobileNetV2 anti-spoof (with LBP fallback)
        self.anti_spoof = AntiSpoofDetector()

        # Thesis metrics logger
        self.metrics = MetricsLogger(log_dir=PROJECT_DIR)

        self.bt_server = BluetoothServer()
        if not self.bt_server.start():
            print("[Main] Bluetooth server failed to start, continuing without BT")

        # 433MHz RF receiver — two-button edge-detection with pulse timing
        self.rf_receiver = RFReceiver(lock_pin=22, unlock_pin=23)
        self.rf_receiver.set_callback(self._handle_rf_command)
        if self.rf_receiver.start():
            print("[Main] RF receiver active — LOCK=GPIO22 UNLOCK=GPIO23")
        else:
            print("[Main] RF receiver unavailable — 433MHz disabled")

        print("[Main] All controllers initialized")
        return True

    # ── Cleanup ──────────────────────────────────────────────────────
    def cleanup(self):
        """Gracefully shut down all controllers."""
        print("[Main] Cleaning up...")
        if self.metrics is not None:
            self.metrics.print_summary()
        if self.bt_server:
            try: self.bt_server.stop()
            except Exception: pass
        if self.camera:
            try: self.camera.stop()
            except Exception: pass
        if self.relay:
            try: self.relay.cleanup()
            except Exception: pass
        if self.buzzer:
            try: self.buzzer.cleanup()
            except Exception: pass
        if self.rf_receiver:
            try: self.rf_receiver.stop()
            except Exception: pass
        print("[Main] Cleanup complete")
        try:
            cv2.destroyWindow(CV2_WINDOW)
        except Exception:
            pass

    # ── Preview Window ──────────────────────────────────────────────
    def _show_preview(self, frame, state_label="", info_lines=None):
        """Annotate frame with status info and show in cv2 window."""
        if frame is None or not self._show_preview_enabled:
            return
        display = frame.copy()
        h, w = display.shape[:2]

        # Draw state label at top
        cv2.rectangle(display, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(display, state_label, (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Draw info lines at bottom
        if info_lines:
            y_start = h - 10 - (len(info_lines) * 22)
            for i, line in enumerate(info_lines):
                y = y_start + i * 22
                cv2.putText(display, line, (10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw face detections
        if hasattr(self, '_last_face_locations') and self._last_face_locations:
            for face_dict in self._last_face_locations:
                x1, y1, x2, y2 = face_dict['bbox']
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(CV2_WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[Main] 'q' pressed — shutting down")
            self._running = False

    # ── Helper: decode base64 image to np.ndarray (BGR) ────────────
    @staticmethod
    def _b64_to_bgr(b64_str):
        """Decode a base64 JPEG string to a BGR numpy array."""
        img_bytes = base64.b64decode(b64_str)
        pil_image = Image.open(io.BytesIO(img_bytes))
        rgb = np.array(pil_image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    # ── RF Remote Command Handler (motor pulses) ────────────────────
    def _handle_rf_command(self, action: str):
        """Called by RFReceiver on LOCK or UNLOCK button press."""
        if action == "LOCK":
            print("[RF] Remote LOCK — motor CCW 1s")
            self.relay.lock_ccw(duration=1.0)
            self._lock_state = "locked"
            self.buzzer.success_beep()
            self.logger.log_event(
                face_id="rf_remote", result="MANUAL_LOCK",
                details="motor CCW 1s via 433MHz"
            )
        elif action == "UNLOCK":
            print("[RF] Remote UNLOCK — motor CW 1s")
            self.relay.unlock_cw(duration=1.0)
            self._lock_state = "unlocked"
            self.buzzer.success_beep()
            self.logger.log_event(
                face_id="rf_remote", result="MANUAL_UNLOCK",
                details="motor CW 1s via 433MHz"
            )

    # ── Bluetooth Command Handler ────────────────────────────────────
    def _handle_bt_command(self, command):
        """Process a BT command dict and return a response dict."""
        action = command.get('cmd', command.get('action', '')).upper()
        print(f"[BT] Received command: {action}")

        if action == 'PING':
            return {'status': 'OK', 'response': 'pong'}

        elif action == 'REGISTER_IMAGE':
            face_id = command.get('face_id', '')
            image_b64 = command.get('image', '')
            if not face_id or not image_b64:
                return {'status': 'ERROR', 'message': 'Missing face_id or image'}

            if self.face_storage.get_face_count() >= FaceStorage.MAX_FACES:
                return {'status': 'ERROR', 'message': f'Maximum {FaceStorage.MAX_FACES} faces reached'}

            try:
                img = self._b64_to_bgr(image_b64)
                result = self.face_recognizer.get_face_encoding(img)
                if result is not None:
                    enc, _ = result
                    if face_id not in self._pending_encodings:
                        self._pending_encodings[face_id] = []
                    self._pending_encodings[face_id].append(enc)
                    idx = len(self._pending_encodings[face_id])
                    print(f"[BT]   Image {idx}/10 for {face_id}")
                    return {'status': 'OK', 'index': idx, 'total': 10}
                else:
                    return {'status': 'ERROR', 'message': 'No face detected in image'}
            except Exception as e:
                print(f"[BT] REGISTER_IMAGE error: {e}")
                traceback.print_exc()
                return {'status': 'ERROR', 'message': str(e)}

        elif action == 'REGISTER_FINALIZE':
            face_id = command.get('face_id', '')
            if not face_id:
                return {'status': 'ERROR', 'message': 'Missing face_id'}

            try:
                encodings = self._pending_encodings.pop(face_id, [])
                if not encodings:
                    return {'status': 'ERROR', 'message': 'No images registered for this face'}

                # Average all encodings
                avg = np.mean(encodings, axis=0)
                avg = avg / (np.linalg.norm(avg) + 1e-6)

                # Pad to exactly 10 for storage
                all_encodings = encodings[:]
                while len(all_encodings) < 10:
                    all_encodings.append(avg)

                self.face_storage.add_face(face_id, all_encodings[:10])
                print(f"[BT] Registered face: {face_id} ({len(encodings)} images)")
                return {'status': 'OK', 'message': f'Face {face_id} registered'}
            except Exception as e:
                print(f"[BT] REGISTER_FINALIZE error: {e}")
                traceback.print_exc()
                return {'status': 'ERROR', 'message': str(e)}

        elif action == 'DELETE':
            face_id = command.get('face_id', '')
            if not face_id:
                return {'status': 'ERROR', 'message': 'Missing face_id'}
            try:
                self.face_storage.delete_face(face_id)
                print(f"[BT] Deleted face: {face_id}")
                return {'status': 'OK', 'message': f'Face {face_id} deleted'}
            except Exception as e:
                return {'status': 'ERROR', 'message': str(e)}

        elif action == 'LIST':
            try:
                faces = self.face_storage.list_faces()
                face_list = []
                for fid, fdata in faces.items():
                    face_list.append({
                        'face_id': fid,
                        'created_at': fdata.get('created_at', ''),
                        'metadata': fdata.get('metadata', {})
                    })
                return {'status': 'OK', 'faces': face_list, 'count': len(face_list)}
            except Exception as e:
                return {'status': 'ERROR', 'message': str(e)}

        elif action == 'GET_LOG':
            try:
                limit = command.get('limit', 50)
                entries = self.logger.get_logs(limit=limit)
                return {'status': 'OK', 'entries': entries}
            except Exception as e:
                return {'status': 'ERROR', 'message': str(e)}

        elif action == 'LOCK':
            print("[BT] Motor LOCK — CCW 1s")
            self.relay.lock_ccw(duration=1.0)
            self._lock_state = "locked"
            self.buzzer.success_beep()
            self.logger.log_event(
                face_id="bt_remote", result="MANUAL_LOCK",
                details="motor CCW 1s via BT"
            )
            return {'status': 'OK', 'message': 'Door locked'}

        elif action == 'UNLOCK':
            print("[BT] Motor UNLOCK — CW 1s")
            self.relay.unlock_cw(duration=1.0)
            self._lock_state = "unlocked"
            self.buzzer.success_beep()
            self.logger.log_event(
                face_id="bt_remote", result="MANUAL_UNLOCK",
                details="motor CW 1s via BT"
            )
            return {'status': 'OK', 'message': 'Door unlocked'}

        elif action == 'GET_STATUS':
            return {
                'status': 'OK',
                'door_state': self._lock_state,
                'rf_configured': True,
                'face_count': self.face_storage.get_face_count() if self.face_storage else 0,
            }

        elif action == 'GET_METRICS':
            if self.metrics:
                return {'status': 'OK', 'metrics': self.metrics.summary()}
            return {'status': 'ERROR', 'message': 'Metrics not available'}

        else:
            return {'status': 'ERROR', 'message': f'Unknown action: {action}'}

    def _process_bt_client(self):
        """Non-blocking check for BT client messages and respond."""
        if not self.bt_server:
            return
        try:
            if not self.bt_server.is_client_connected():
                return
            cmd = self.bt_server.receive(timeout=0.0)
            if cmd is not None and isinstance(cmd, dict):
                response = self._handle_bt_command(cmd)
                self.bt_server.send(response)
        except Exception as e:
            print(f"[Main] BT processing error: {e}")

    # ── Helper: crop face from frame ─────────────────────────────────
    def _crop_face(self, frame: np.ndarray, bbox) -> np.ndarray:
        """Crop face region from frame using bbox [x1, y1, x2, y2]."""
        x1 = max(0, int(bbox[0]))
        y1 = max(0, int(bbox[1]))
        x2 = min(frame.shape[1], int(bbox[2]))
        y2 = min(frame.shape[0], int(bbox[3]))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return frame  # fallback: whole frame
        return crop

    # ── State: SCANNING ──────────────────────────────────────────────
    def _state_scanning(self):
        """Capture frames, detect faces, listen for BT."""
        frame = self.camera.capture_frame()
        if frame is None:
            time.sleep(FRAME_INTERVAL)
            return

        self._latest_frame = frame
        self._frame_count += 1
        self._fps_counter += 1

        # Log FPS every 30 frames
        if self._fps_counter >= 30:
            elapsed = time.time() - self._fps_timer
            fps = self._fps_counter / elapsed
            print(f"[Main] ~{fps:.1f} fps ({self._frame_count} frames)")
            self._fps_timer = time.time()
            self._fps_counter = 0

        # Face detection every 3rd frame only (saves ~2x speed)
        if self._frame_count % 3 == 1:
            face_locations = self.face_recognizer.detect_faces(frame)
            self._last_face_locations = face_locations
        else:
            face_locations = self._last_face_locations or []

        # Show preview
        lock_icon = "\U0001f512" if self._lock_state == "locked" else "\U0001f513"
        if face_locations:
            self._show_preview(frame, f"SCANNING — FACE DETECTED {lock_icon}",
                               [f"Faces: {len(face_locations)}", f"Door: {self._lock_state.upper()}"])
        else:
            self._show_preview(frame, f"SCANNING {lock_icon}",
                               [f"Door: {self._lock_state.upper()}"])

        if face_locations:
            # Log detection quality: face count, size, position
            _largest = max(face_locations,
                           key=lambda f: (f['bbox'][2]-f['bbox'][0])*(f['bbox'][3]-f['bbox'][1]))
            _fw = _largest['bbox'][2] - _largest['bbox'][0]
            _fh = _largest['bbox'][3] - _largest['bbox'][1]
            print(f"[Main] Face detected ({len(face_locations)} found, "
                  f"largest: {_fw}x{_fh}px at ({_largest['bbox'][0]},{_largest['bbox'][1]}))")
            print("[Main] Transitioning: SCANNING → COMPARE (anti-spoof + ArcFace)")
            self.state = State.COMPARE
            return

        # Non-blocking BT
        if self.bt_server and not self.bt_server.is_client_connected():
            try:
                if self.bt_server.wait_for_connection(timeout=0):
                    print("[Main] BT client connected")
            except Exception:
                pass

        self._process_bt_client()

    # ── State: COMPARE (anti-spoof + ArcFace recognition) ───────────
    def _state_compare(self):
        """Single-frame pipeline: anti-spoof → ArcFace encoding → match."""
        print("[Main] COMPARE: single-frame anti-spoof + ArcFace recognition")
        t_start = time.perf_counter()

        if self._latest_frame is None:
            print("[Main] No frame to compare, returning to SCANNING")
            self.state = State.SCANNING
            return

        frame = self._latest_frame

        try:
            # ── Single call: RetinaFace detects + ArcFace encodes ─────
            t1 = time.perf_counter()
            faces = self.face_recognizer.detect_faces(frame)
            t_detect_ms = (time.perf_counter() - t1) * 1000

            if not faces:
                print("[Main] No face in compare frame")
                self.state = State.SCANNING
                return

            largest_face = max(
                faces,
                key=lambda f: (f['bbox'][2]-f['bbox'][0]) *
                              (f['bbox'][3]-f['bbox'][1]),
            )
            bbox = largest_face['bbox']
            embedding = largest_face['embedding']
            det_score = largest_face['det_score']
            face_crop = self._crop_face(frame, bbox)
            print(f"[Main]   RetinaFace: {len(faces)} face(s), "
                  f"largest at ({bbox[0]},{bbox[1]}-{bbox[2]},{bbox[3]}), "
                  f"conf={det_score:.2f}, crop={face_crop.shape[1]}x{face_crop.shape[0]}px")

            # ── Stage 2: Anti-Spoof ──────────────────────────────────
            t1 = time.perf_counter()
            live_score = self.anti_spoof.predict(face_crop)
            t_anti_spoof_ms = (time.perf_counter() - t1) * 1000

            is_live = live_score >= LIVE_THRESHOLD
            print(f"[Main]   Anti-spoof: score={live_score:.3f} "
                  f"(threshold={LIVE_THRESHOLD}) {'LIVE ✅' if is_live else 'SPOOF ❌'}")

            if not is_live:
                print("[Main]   REJECTED — spoof detected")
                self.metrics.log_frame(
                    detect_ms=t_detect_ms,
                    anti_spoof_ms=t_anti_spoof_ms,
                    encode_ms=0,
                    match_ms=0,
                    anti_spoof_score=live_score,
                    is_live=False,
                    result="REJECTED",
                )
                self._last_score = live_score
                self.state = State.REJECTED
                return

            # ── Stage 3: Encoding (already from RetinaFace) ──────
            t_encode_ms = 0.0  # embedding included in detect_faces call

            # ── Stage 4: Matching (cosine sim for ArcFace 512-D) ──
            t1 = time.perf_counter()
            stored_faces = self.face_storage.list_faces()
            best_match = None
            best_sim = -1.0

            if stored_faces:
                for face_id, face_data in stored_faces.items():
                    for stored_enc in face_data.get('encoding', []):
                        # Cosine similarity (ArcFace embeddings are L2-normed)
                        sim = float(np.dot(embedding, stored_enc))
                        if sim > best_sim:
                            best_sim = sim
                            best_match = face_id

            t_match_ms = (time.perf_counter() - t1) * 1000

            # ── Decision ─────────────────────────────────────────────
            granted = best_match is not None and best_sim >= MATCH_THRESHOLD

            if granted:
                print(f"[Main]   Match: {best_match} (cos sim={best_sim:.4f}) — GRANTED")
                self._matched_id = best_match
                self._last_distance = 1.0 - best_sim  # store as dissimilarity
                self._show_preview(frame, f"MATCH: {best_match} ✅",
                                   [f"Score: {best_sim:.3f}",
                                    f"Threshold: {MATCH_THRESHOLD}"])
                self.state = State.GRANTED
            else:
                reason = "no_stored_faces" if not stored_faces else f"sim={best_sim:.4f}"
                print(f"[Main]   No match ({reason}) — REJECTED")
                self._show_preview(frame, "NO MATCH ❌",
                                   [f"Best similarity: {best_sim:.3f}",
                                    f"Threshold: {MATCH_THRESHOLD}"])
                self.state = State.REJECTED

            # ── Log metrics ──────────────────────────────────────────
            self.metrics.log_frame(
                detect_ms=t_detect_ms,
                anti_spoof_ms=t_anti_spoof_ms,
                encode_ms=t_encode_ms,
                match_ms=t_match_ms,
                anti_spoof_score=live_score,
                is_live=True,
                match_id=best_match if granted else None,
                match_distance=1.0 - best_sim if not granted else None,
                result="GRANTED" if granted else "REJECTED",
            )

        except Exception as e:
            print(f"[Main] Compare error: {e}")
            traceback.print_exc()
            self.state = State.REJECTED

    # ── State: GRANTED ───────────────────────────────────────────────
    def _state_granted(self):
        """Unlock relay, success beep, log event, return to SCANNING."""
        face_id = getattr(self, '_matched_id', 'unknown')
        print(f"[Main] GRANTED — unlocking (CW) {UNLOCK_DURATION}s for {face_id}")
        try:
            self.relay.unlock_cw(duration=UNLOCK_DURATION)
            self.buzzer.success_beep()
        except Exception as e:
            print(f"[Main] Relay/buzzer error: {e}")

        self.logger.log_event(
            face_id=face_id,
            result='GRANTED',
            details=f'distance={self._last_distance:.4f}'
        )

        if self._latest_frame is not None:
            for _ in range(int(UNLOCK_DURATION * 5)):
                if not self._running:
                    break
                self._show_preview(self._latest_frame, f"DOOR UNLOCKED — {face_id} ✅")
                time.sleep(0.2)

        self.state = State.SCANNING

    # ── State: REJECTED ──────────────────────────────────────────────
    def _state_rejected(self):
        """Fail beeps, log event, return to SCANNING."""
        print("[Main] REJECTED — fail beep")
        try:
            self.buzzer.fail_beep()
        except Exception as e:
            print(f"[Main] Buzzer error: {e}")

        self.logger.log_event(
            face_id='unknown',
            result='REJECTED',
            details=f'anti_spoof_spoof_or_no_match'
        )

        if self._latest_frame is not None:
            self._show_preview(self._latest_frame, "ACCESS DENIED ❌")
            time.sleep(1)

        self.state = State.SCANNING

    # ── Main Loop ────────────────────────────────────────────────────
    def run(self):
        """Main state machine loop."""
        print("[Main] Face Door System starting (ArcFace + MobileNetV2 anti-spoof)...")

        if not self._init_controllers():
            self.cleanup()
            return

        self.state = State.SCANNING
        print(f"[Main] Entering auto-scan loop at ~{FRAME_RATE}fps")

        while self._running:
            loop_start = time.perf_counter()

            try:
                if self.state == State.SCANNING:
                    self._state_scanning()
                elif self.state == State.COMPARE:
                    self._state_compare()
                elif self.state == State.GRANTED:
                    self._state_granted()
                elif self.state == State.REJECTED:
                    self._state_rejected()
            except Exception as e:
                print(f"[Main] Unhandled error in state {self.state}: {e}")
                traceback.print_exc()
                self.state = State.SCANNING
                continue

            # Maintain framerate
            elapsed = time.perf_counter() - loop_start
            sleep_time = FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Process BT messages during all states
            self._process_bt_client()

        self.cleanup()
        print("[Main] System stopped")


# ── Entry Point ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    system = FaceDoorSystem()
    system.run()
