#!/usr/bin/env python3
"""
Main entry point for the face recognition door system.
State-machine-driven auto-scan loop with Bluetooth command handling.
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
    from ir_liveness import IRLivenessDetector
except ImportError as e:
    print(f"[Main] FATAL: Could not import sibling modules: {e}")
    print("[Main] Make sure all modules exist in", PROJECT_DIR)
    sys.exit(1)

from bluetooth_server import BluetoothServer
from logger import ActivityLogger


# ── Constants ───────────────────────────────────────────────────────────
FRAME_RATE = 15.0
FRAME_INTERVAL = 1.0 / FRAME_RATE
MAX_COLLECT_FRAMES = 15        # reserved for potential multi-frame scoring
UNLOCK_DURATION = 1.0
MATCH_THRESHOLD = 0.6


# ── State Machine ──────────────────────────────────────────────────────
class State:
    INIT = "INIT"
    SCANNING = "SCANNING"
    COLLECTING = "COLLECTING"
    LIVENESS_CHECK = "LIVENESS_CHECK"
    COMPARE = "COMPARE"
    GRANTED = "GRANTED"
    REJECTED = "REJECTED"


# CV2 display window name
CV2_WINDOW = "Face Door System"

# ── CLI args ─────────────────────────────────────────────────────────
HEADLESS = "--headless" in sys.argv or "-H" in sys.argv
PREVIEW = "--preview" in sys.argv or "-p" in sys.argv


class FaceDoorSystem:
    """Main door system orchestrator with state machine loop."""

    def __init__(self):
        self.state = State.INIT
        self._running = True
        self._liveness_frames = []
        self._latest_frame = None
        self._matched_id = None
        self._last_distance = float('inf')
        self._show_preview_enabled = False
        self._frame_count = 0
        self._fps_timer = time.time()
        self._fps_counter = 0  # will be set after GUI check

        self.camera = None
        self.relay = None
        self.buzzer = None
        self.face_storage = None
        self.face_recognizer = None
        self.ir_liveness = None
        self.bt_server = None
        self.logger = None
        self.rf_receiver = None

        # Persistent lock state (RF remote / BT manual override)
        self._lock_state = "locked"  # "locked" | "unlocked"
        self._ir_face_rect = None

        # RF learn mode
        self._rf_learning = None
        self._rf_learn_step = None
        self._rf_learn_timeout = 0

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

        # Preview window — only if a display is available (HDMI monitor or ssh -X)
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

        self.relay = RelayController()
        self.buzzer = BuzzerController()
        self.face_storage = FaceStorage()
        self.face_recognizer = FaceRecognizer()
        self.logger = ActivityLogger()

        # NoIR spectral liveness detection (single-frame, no external LED needed)
        self.ir_liveness = IRLivenessDetector()

        self.bt_server = BluetoothServer()
        if not self.bt_server.start():
            print("[Main] Bluetooth server failed to start, continuing without BT")

        # 433MHz RF receiver for remote lock/unlock
        self.rf_receiver = RFReceiver(pin=22)
        self.rf_receiver.set_callback(self._handle_rf_command)
        if self.rf_receiver.start():
            if self.rf_receiver.is_configured:
                print(f"[Main] RF receiver active — remote codes loaded")
            else:
                print("[Main] RF receiver active — no codes configured yet")
        else:
            print("[Main] RF receiver unavailable — 433MHz disabled")

        print("[Main] All controllers initialized")
        return True

    # ── Cleanup ──────────────────────────────────────────────────────
    def cleanup(self):
        """Gracefully shut down all controllers."""
        print("[Main] Cleaning up...")
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
            for rect in self._last_face_locations:
                x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow(CV2_WINDOW, display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[Main] 'q' pressed — shutting down")
            self._running = False
        elif key == ord('d'):
            # Toggle debug overlay with distance/match info
            self._show_debug = not getattr(self, '_show_debug', False)

    # ── Helper: decode base64 image to np.ndarray (BGR) ────────────
    @staticmethod
    def _b64_to_bgr(b64_str):
        """Decode a base64 JPEG string to a BGR numpy array."""
        img_bytes = base64.b64decode(b64_str)
        pil_image = Image.open(io.BytesIO(img_bytes))
        rgb = np.array(pil_image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    # ── RF Remote Command Handler ────────────────────────────────────
    def _handle_rf_command(self, action: str):
        """Called by RFReceiver when a known remote code arrives."""
        if action == "LOCK":
            print("[RF] Remote LOCK command")
            self.relay.lock_persistent()
            self._lock_state = "locked"
            self.buzzer.success_beep()
            self.logger.log_event(
                face_id="rf_remote", result="MANUAL_LOCK",
                details="via 433MHz remote"
            )
        elif action == "UNLOCK":
            print("[RF] Remote UNLOCK command")
            self.relay.unlock_persistent()
            self._lock_state = "unlocked"
            self.buzzer.success_beep()
            self.logger.log_event(
                face_id="rf_remote", result="MANUAL_UNLOCK",
                details="via 433MHz remote"
            )

    # ── RF Learn Callback ─────────────────────────────────────────
    def _rf_learn_callback(self, code: int, pulselen: int):
        """
        Called during learn mode when any RF code is received.
        Automatically walks through: LOCK step → UNLOCK step → done.
        """
        action = self._rf_learn_step
        print(f"[RF] Learn: received code {code} (pulselen={pulselen}) as {action}")

        if action == "lock":
            self.rf_receiver.save_code(code, pulselen, "lock")
            self._rf_learn_step = "unlock"
            self._rf_learn_timeout = time.time() + 30
            print("[RF] Learn: LOCK saved — now press UNLOCK button")
        elif action == "unlock":
            self.rf_receiver.save_code(code, pulselen, "unlock")
            self.rf_receiver.set_learn_callback(None)
            self._rf_learning = None
            self._rf_learn_step = None
            print("[RF] Learn complete — LOCK and UNLOCK codes saved")

    # ── Bluetooth Command Handler ────────────────────────────────────
    def _handle_bt_command(self, command):
        """Process a BT command dict and return a response dict."""
        action = command.get('cmd', command.get('action', '')).upper()
        print(f"[BT] Received command: {action}")

        if action == 'PING':
            return {'status': 'OK', 'response': 'pong'}

        elif action == 'REGISTER':
            face_id = command.get('face_id', '')
            images_b64 = command.get('images', [])
            if not face_id or not images_b64:
                return {'status': 'ERROR', 'message': 'Missing face_id or images'}

            # Check capacity early
            if self.face_storage.get_face_count() >= FaceStorage.MAX_FACES:
                return {'status': 'ERROR', 'message': f'Maximum {FaceStorage.MAX_FACES} faces reached'}

            try:
                # Decode base64 images to numpy arrays
                images = [self._b64_to_bgr(b64) for b64 in images_b64]

                # Register face: get averaged encoding from all valid images
                avg_encoding = self.face_recognizer.register_face(images)
                if avg_encoding is None:
                    return {'status': 'ERROR', 'message': 'No face detected in any image'}

                # Collect all 10 individual encodings for storage
                all_encodings = []
                for img in images:
                    result = self.face_recognizer.get_face_encoding(img)
                    if result is not None:
                        enc, _ = result
                        all_encodings.append(enc)
                    else:
                        all_encodings.append(avg_encoding)  # fallback

                # Pad to exactly 10 if some images had no face
                while len(all_encodings) < 10:
                    all_encodings.append(avg_encoding)

                # Store all 10 encodings
                self.face_storage.add_face(face_id, all_encodings[:10])
                print(f"[BT] Registered face: {face_id}")
                return {'status': 'OK', 'message': f'Face {face_id} registered'}

            except Exception as e:
                print(f"[BT] Registration error: {e}")
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
                # Return metadata only (not raw encodings) for listing
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

        # ── Manual Lock / Unlock (persistent state) ─────────────────
        elif action == 'LOCK_MANUAL' or action == 'LOCK':
            print("[BT] Manual LOCK command")
            self.relay.lock_persistent()
            self._lock_state = "locked"
            self.buzzer.success_beep()
            self.logger.log_event(
                face_id="bt_remote", result="MANUAL_LOCK",
                details="via Bluetooth"
            )
            return {'status': 'OK', 'message': 'Door locked'}

        elif action == 'UNLOCK_MANUAL' or action == 'UNLOCK':
            print("[BT] Manual UNLOCK command")
            self.relay.unlock_persistent()
            self._lock_state = "unlocked"
            self.buzzer.success_beep()
            self.logger.log_event(
                face_id="bt_remote", result="MANUAL_UNLOCK",
                details="via Bluetooth"
            )
            return {'status': 'OK', 'message': 'Door unlocked'}

        elif action == 'GET_STATUS':
            return {
                'status': 'OK',
                'door_state': self._lock_state,
                'rf_configured': self.rf_receiver.is_configured if self.rf_receiver else False,
                'face_count': self.face_storage.get_face_count() if self.face_storage else 0,
            }

        # ── RF Learn Mode ──────────────────────────────────────────
        elif action == 'LEARN_RF':
            if not self.rf_receiver or not self.rf_receiver._RF_AVAILABLE:
                return {'status': 'ERROR', 'message': 'RF receiver not available'}
            self._rf_learning = {}
            self._rf_learn_step = "lock"
            self.rf_receiver.set_learn_callback(self._rf_learn_callback)
            print("[BT] RF learn mode started — press LOCK button on remote")
            # Schedule timeout to abort learn mode after 30s
            self._rf_learn_timeout = time.time() + 30
            return {
                'status': 'OK',
                'message': 'Learn mode started. Press LOCK button on your remote.',
                'step': 'lock'
            }

        elif action == 'SAVE_RF':
            code = command.get('code')
            pulselen = command.get('pulselen')
            action_to_save = command.get('action', '').lower()
            if not code or action_to_save not in ('lock', 'unlock'):
                return {'status': 'ERROR', 'message': 'Missing code or invalid action'}
            self.rf_receiver.save_code(int(code), int(pulselen), action_to_save)
            self.rf_receiver.set_learn_callback(None)  # exit learn mode
            self._rf_learning = None
            self._rf_learn_step = None
            return {
                'status': 'OK',
                'message': f'RF code {code} saved for {action_to_save}'
            }

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

    # ── State: SCANNING ──────────────────────────────────────────────
    def _state_scanning(self):
        """Capture frames, detect faces (every 3rd frame), listen for BT."""
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
        lock_icon = "🔒" if self._lock_state == "locked" else "🔓"
        if face_locations:
            self._show_preview(frame, f"SCANNING — FACE DETECTED {lock_icon}",
                               [f"Faces: {len(face_locations)}", f"Door: {self._lock_state.upper()}"])
        else:
            self._show_preview(frame, f"SCANNING {lock_icon}",
                               [f"Door: {self._lock_state.upper()}"])

        if face_locations:
            print("[Main] Face detected — NoIR spectral liveness check via COLLECTING")
            # Store the largest face rect for spectral crop
            self._ir_face_rect = max(face_locations,
                                     key=lambda r: (r.right()-r.left())*(r.bottom()-r.top()))
            self._liveness_frames = []  # reset for spectral capture
            self.state = State.COLLECTING
            return

        # Non-blocking BT client accept
        if self.bt_server and not self.bt_server.is_client_connected():
            try:
                if self.bt_server.wait_for_connection(timeout=0):
                    print("[Main] BT client connected")
            except Exception:
                pass

        self._process_bt_client()

    # ── State: COLLECTING (single-frame for NoIR spectral analysis) ──
    def _state_collecting(self):
        """Capture a single frame for NoIR spectral liveness detection.

        Uses the Pi NoIR camera's inherent IR sensitivity (no IR-cut filter)
        to analyze the spectral signature of the face region.
        Goes to LIVENESS_CHECK after capture.
        """
        frame = self.camera.capture_frame()
        if frame is None:
            print("[Main] COLLECTING: frame capture failed")
            self.state = State.SCANNING
            return

        self._liveness_frames = [frame]
        self._latest_frame = frame

        print("[Main] Frame captured — running NoIR spectral liveness")
        self._show_preview(frame, "NOIR LIVENESS — ANALYZING",
                           ["Single-frame spectral analysis"])
        self.state = State.LIVENESS_CHECK

    # ── State: LIVENESS_CHECK (NoIR spectral analysis) ────────────────
    def _state_liveness_check(self):
        """Run NoIR spectral liveness analysis on the captured frame.

        Analyzes the R/G/B channel balance in the face region to distinguish
        live skin (R channel boosted by NIR bleed) from photos/screens.
        """
        if not self._liveness_frames:
            print("[Main] No frame for liveness analysis — rejecting")
            self.state = State.REJECTED
            return

        frame = self._liveness_frames[0]

        if not hasattr(self, '_ir_face_rect') or self._ir_face_rect is None:
            print("[Main] No face rect for spectral analysis — falling back to COMPARE")
            self.state = State.COMPARE
            return

        print("[Main] Analyzing NoIR spectral liveness...")
        try:
            result = self.ir_liveness.check_liveness(frame, self._ir_face_rect)
            passed = result['passed']
            score = result['score']
            red_dom = result['red_dominance']
            red_excess = result['red_excess']
            details = result.get('details', '')

            print(f"[Main]   R/(G+B)={red_dom:.3f}, red_excess={red_excess:.3f}, score={score:.3f}")

            if passed:
                print(f"[Main] NOIR SPECTRAL LIVENESS PASSED ✅ (score={score:.3f})")
                self._show_preview(frame, "NOIR LIVENESS PASSED ✅",
                                   [f"Score: {score:.3f}", f"R/(G+B): {red_dom:.3f}", details])
                time.sleep(0.5)
                self.state = State.COMPARE
            else:
                print(f"[Main] NOIR SPECTRAL LIVENESS FAILED ❌ (score={score:.3f})")
                self._show_preview(frame, "NOIR LIVENESS FAILED ❌",
                                   [f"Score: {score:.3f}", f"R/(G+B): {red_dom:.3f}", details])
                time.sleep(1)
                self.state = State.REJECTED
        except Exception as e:
            print(f"[Main] NoIR spectral liveness error: {e}")
            traceback.print_exc()
            self.state = State.REJECTED

    # ── State: COMPARE ───────────────────────────────────────────────
    def _state_compare(self):
        """Compare the latest frame encoding against stored faces."""
        print("[Main] Comparing face...")
        if self._latest_frame is None:
            print("[Main] No frame to compare, returning to SCANNING")
            self.state = State.SCANNING
            return

        try:
            result = self.face_recognizer.get_face_encoding(self._latest_frame)
            if result is None:
                print("[Main] Could not extract encoding, rejecting")
                self.state = State.REJECTED
                return

            encoding, _ = result
            stored_faces = self.face_storage.list_faces()

            if not stored_faces:
                print("[Main] No stored faces to compare against, rejecting")
                self.state = State.REJECTED
                return

            # Match against all stored encodings (each face stores 10)
            best_match = None
            self._last_distance = float('inf')
            for face_id, face_data in stored_faces.items():
                for stored_enc in face_data.get('encoding', []):
                    distance = np.linalg.norm(encoding - stored_enc)
                    if distance < self._last_distance:
                        self._last_distance = distance
                        best_match = face_id

            if best_match is not None and self._last_distance < MATCH_THRESHOLD:
                print(f"[Main] Match: {best_match} (dist={self._last_distance:.4f}) — GRANTED")
                self._matched_id = best_match
                self._show_preview(self._latest_frame, f"MATCH: {best_match} ✅",
                                   [f"Distance: {self._last_distance:.3f}",
                                    f"Threshold: {MATCH_THRESHOLD}"])
                self.state = State.GRANTED
            else:
                reason = "no_match" if best_match is None else f"dist={self._last_distance:.4f}"
                print(f"[Main] No match ({reason}) — REJECTED")
                self._show_preview(self._latest_frame, "NO MATCH ❌",
                                   [f"Best dist: {self._last_distance:.3f}",
                                    f"Threshold: {MATCH_THRESHOLD}"])
                time.sleep(1.5)

        except Exception as e:
            print(f"[Main] Compare error: {e}")
            traceback.print_exc()
            self.state = State.REJECTED

    # ── State: GRANTED ───────────────────────────────────────────────
    def _state_granted(self):
        """Unlock relay, success beep, log event, return to SCANNING."""
        face_id = getattr(self, '_matched_id', 'unknown')
        print(f"[Main] GRANTED — unlocking {UNLOCK_DURATION}s for {face_id}")
        try:
            self.relay.unlock(duration=UNLOCK_DURATION)
            self.buzzer.success_beep()
        except Exception as e:
            print(f"[Main] Relay/buzzer error: {e}")

        self.logger.log_event(
            face_id=face_id,
            result='GRANTED',
            details=f'distance={self._last_distance:.4f}'
        )

        # Keep showing the match on screen during unlock
        if self._latest_frame is not None:
            for _ in range(int(UNLOCK_DURATION * 5)):
                if not self._running:
                    break
                self._show_preview(self._latest_frame, f"DOOR UNLOCKED — {face_id} ✅")
                time.sleep(0.2)

        self._liveness_frames = []
        self._ir_face_rect = None
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
            details='liveness_failed_or_no_match'
        )

        # Show rejection on screen briefly
        if self._latest_frame is not None:
            self._show_preview(self._latest_frame, "ACCESS DENIED ❌")
            time.sleep(1)

        self._liveness_frames = []
        self._ir_face_rect = None
        self.state = State.SCANNING

    # ── Main Loop ────────────────────────────────────────────────────
    def run(self):
        """Main state machine loop."""
        print("[Main] Face Door System starting...")

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
                elif self.state == State.COLLECTING:
                    self._state_collecting()
                elif self.state == State.LIVENESS_CHECK:
                    self._state_liveness_check()
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
                self._liveness_frames = []
                continue

            # Maintain framerate
            elapsed = time.perf_counter() - loop_start
            sleep_time = FRAME_INTERVAL - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            # Process BT messages during all states
            self._process_bt_client()

            # RF learn mode timeout check
            if self._rf_learn_step is not None:
                if time.time() > self._rf_learn_timeout:
                    print("[Main] RF learn mode timed out")
                    if self.rf_receiver:
                        self.rf_receiver.set_learn_callback(None)
                    self._rf_learning = None
                    self._rf_learn_step = None
                    self._rf_learn_timeout = 0

        self.cleanup()
        print("[Main] System stopped")


# ── Entry Point ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    system = FaceDoorSystem()
    system.run()
