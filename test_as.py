"""Capture with photo, then wait (5s) for user to remove it, capture real face."""
import cv2, numpy as np, onnxruntime as ort, time
from picamera2 import Picamera2

session = ort.InferenceSession("models/minifasnet_v2.onnx", providers=["CPUExecutionProvider"])

picam = Picamera2()
config = picam.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 15},
)
picam.configure(config)
picam.start()
time.sleep(2)

print("=== HOLD PHOTO NOW ===")
time.sleep(3)

# Capture with photo
for _ in range(8):
    frame_photo = picam.capture_array("main")
    time.sleep(0.2)

print("=== REMOVE PHOTO (show real face) ===")
time.sleep(5)

# Capture real face
for _ in range(8):
    frame_live = picam.capture_array("main")
    time.sleep(0.2)

picam.stop()
picam.close()

import insightface
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"], root="~/.insightface")
app.prepare(ctx_id=0, det_size=(320, 320))

def classify(frame, label):
    faces = app.get(frame)
    if len(faces) == 0:
        print(f"[{label}] No face detected")
        return
    x1,y1,x2,y2 = faces[0].bbox.astype(int)
    face_bgr = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
    print(f"[{label}] Face at ({x1},{y1})-({x2},{y2}), crop={x2-x1}x{y2-y1}")
    
    # Preprocess as [0,255] (correct)
    img = cv2.resize(face_bgr, (80, 80))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = np.transpose(img, (2, 0, 1))[None, :]
    
    raw = session.run(None, {"input": img})[0][0]
    exp = np.exp(raw - np.max(raw))
    sm = exp / np.sum(exp)
    print(f"[{label}] Raw: {raw}")
    print(f"[{label}] Softmax: {sm}")
    print(f"[{label}] Class 0: {sm[0]:.4f}, Class 1: {sm[1]:.4f}, Class 2: {sm[2]:.4f}")
    print(f"[{label}] Argmax: {np.argmax(sm)}")
    verdict = "SPOOF" if np.argmax(sm) != 1 else "LIVE"
    print(f"[{label}] If Live=class1: {verdict}")
    verdict2 = "SPOOF" if np.argmax(sm) != 2 else "LIVE"
    print(f"[{label}] If Live=class2: {verdict2}")

classify(frame_photo, "PHOTO")
classify(frame_live, "LIVE")
