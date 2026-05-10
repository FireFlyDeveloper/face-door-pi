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
    from liveness_detector import LivenessDetector
except ImportError as e:
    print(f"[Main] FATAL: Could not import sibling modules: {e}")
    print("[Main] Make sure all modules exist in", PROJECT_DIR)
    sys.exit(1)

from bluetooth_server import BluetoothServer
from logger import ActivityLogger


# ── Constants ───────────────────────────────────────────────────────────
FRAME_RATE = 15.0
FRAME_INTERVAL = 1.0 / FRAME_RATE
LIVENESS_FRAMES = 30
UNLOCK_DURATION = 3.0
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
        self.liveness = None
        self.bt_server = None
        self.logger = None

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
        self.liveness = LivenessDetector()
        self.logger = ActivityLogger()

        self.bt_server = BluetoothServer()
        if not self.bt_server.start():
            print("[Main] Bluetooth server failed to start, continuing without BT")

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
        if face_locations:
            self._show_preview(frame, "SCANNING — FACE DETECTED",
                               [f"Faces: {len(face_locations)}"])
        else:
            self._show_preview(frame, "SCANNING")

        if face_locations:
            print("[Main] Face detected — transitioning to COLLECTING")
            self._liveness_frames = []
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

    #    ── State: COLLECTING ────────────────────────────────────────────
    def _state_collecting(self):
        """Collect LIVENESS_FRAMES frames for liveness, abort if face lost."""
        frame = self.camera.capture_frame()
        if frame is None:
            return

        self._latest_frame = frame
        self._liveness_frames.append(frame)
        self._frame_count += 1

        # Every 5 frames, verify face is still visible
        if len(self._liveness_frames) % 5 == 0:
            faces = self.face_recognizer.detect_faces(frame)
            self._last_face_locations = faces
            if not faces:
                print("[Main] Face lost during collection — resetting to SCANNING")
                self._liveness_frames = []
                self.state = State.SCANNING
                return

        # Show preview with progress
        progress = len(self._liveness_frames)
        self._show_preview(frame, "COLLECTING",
                           [f"Frame {progress}/{LIVENESS_FRAMES}"])

        if len(self._liveness_frames) >= LIVENESS_FRAMES:
            print(f"[Main] Collected {LIVENESS_FRAMES} frames for liveness")
            self.state = State.LIVENESS_CHECK

    # ── State: LIVENESS_CHECK ────────────────────────────────────────
    def _state_liveness_check(self):
        """Run 3-layer liveness detection on collected frames."""
        print("[Main] Running liveness check...")
        try:
            result = self.liveness.check_liveness(self._liveness_frames)
            score = result['score']
            blink_s = result.get('blink_score', 0)
            head_pose_s = result.get('head_pose_score', 0)
            head_trans_s = result.get('head_trans_score', 0)
            screen_s = result.get('screen_score', 0)
            head_s = result.get('head_score', 0)
            details = result.get('details', '')
            print(f"[Main]   Blink={blink_s:.2f}  HeadRot={head_pose_s:.2f}  Screen={screen_s:.2f}  Combined={score:.2f}")
            if result['passed']:
                print(f"[Main] Liveness PASSED (score={score:.3f})")
                self._show_preview(self._latest_frame, "LIVENESS PASSED ✅",
                                   [f"Score: {score:.2f}", details])
                self.state = State.COMPARE
            else:
                print(f"[Main] Liveness FAILED (score={score:.3f})")
                self._show_preview(self._latest_frame, "LIVENESS FAILED ❌",
                                   [f"Score: {score:.2f}", details])
                time.sleep(1)
                self.state = State.REJECTED
        except Exception as e:
            print(f"[Main] Liveness check error: {e}")
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

        self.cleanup()
        print("[Main] System stopped")


# ── Entry Point ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    system = FaceDoorSystem()
    system.run()
