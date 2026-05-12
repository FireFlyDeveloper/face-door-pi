"""Test MiniFASNet with correct [0,255] preprocessing and verify class mapping."""
import cv2
import numpy as np
import onnxruntime as ort
import time
import sys

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

# Capture 5 frames to settle
for i in range(5):
    frame = picam.capture_array("main")
    time.sleep(0.3)

picam.stop()
picam.close()

print(f"Frame: {frame.shape}, range [{frame.min()}, {frame.max()}]")

import insightface
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"], root="~/.insightface")
app.prepare(ctx_id=0, det_size=(320, 320))
faces = app.get(frame)

if len(faces) == 0:
    print("NO FACE DETECTED")
    sys.exit(0)

x1, y1, x2, y2 = faces[0].bbox.astype(int)
face_bgr = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)

print(f"\n=== Real face (no photo) ===")
# Preprocess as [0,255] (CORRECT MiniFASNet)
img = cv2.resize(face_bgr, (80, 80))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)
img = np.transpose(img, (2, 0, 1))[None, :]

outputs = session.run(None, {"input": img})
raw = outputs[0][0]
exp = np.exp(raw - np.max(raw))
sm = exp / np.sum(exp)

print(f"  Raw: {raw}")
print(f"  Softmax: {sm}")
print(f"  Index 0: {sm[0]:.4f}  (spoof?)")
print(f"  Index 1: {sm[1]:.4f}  (replay?)")
print(f"  Index 2: {sm[2]:.4f}  (live?)")
