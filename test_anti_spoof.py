"""Test MiniFASNet with [0,255] input range (not [0,1])."""
import cv2
import numpy as np
import onnxruntime as ort
import logging
import sys
import time

logging.basicConfig(level=logging.DEBUG)

from picamera2 import Picamera2

picam = Picamera2()
config = picam.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 15},
)
picam.configure(config)
picam.start()
time.sleep(2)

for i in range(3):
    frame = picam.capture_array("main")
    time.sleep(0.5)

picam.stop()
picam.close()

print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}, range: [{frame.min()}, {frame.max()}]")

import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"], root="~/.insightface")
app.prepare(ctx_id=0, det_size=(320, 320))
faces = app.get(frame)

if len(faces) == 0:
    print("No faces detected")
    sys.exit(0)

print(f"Found {len(faces)} faces")
x1, y1, x2, y2 = faces[0].bbox.astype(int)
face_bgr = frame[y1:y2, x1:x2]  # picamera2 gives RGB, but stay consistent
# Actually frame is RGB from picamera2, but MiniFASNet expects BGR -> RGB conversion internally
# Let's convert properly: what the original code does is take BGR input and convert to RGB
# Picamera2 gives RGB, so for MiniFASNet we need to convert RGB->BGR first, then the predict code converts BGR->RGB
face_bgr = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)

session = ort.InferenceSession("models/minifasnet_v2.onnx", providers=["CPUExecutionProvider"])

# Test 1: [0, 255] range (original MiniFASNet expects this)
img = cv2.resize(face_bgr, (80, 80))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MiniFASNet expects RGB
img = img.astype(np.float32)  # NO /255!
img = np.transpose(img, (2, 0, 1))[None, :]

outputs = session.run(None, {"input": img})
print(f"\nTest [0,255]:")
print(f"  Raw: {outputs[0]}")
scores = outputs[0][0]
exp = np.exp(scores - np.max(scores))
sm = exp / np.sum(exp)
print(f"  Softmax: {sm}")
print(f"  Live(idx2): {sm[2]:.4f}")

# Test 2: [0, 1] normalized (what current code does)
img2 = cv2.resize(face_bgr, (80, 80))
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2 = img2.astype(np.float32) / 255.0
img2 = np.transpose(img2, (2, 0, 1))[None, :]

outputs2 = session.run(None, {"input": img2})
print(f"\nTest [0,1]:")
print(f"  Raw: {outputs2[0]}")
scores2 = outputs2[0][0]
exp2 = np.exp(scores2 - np.max(scores2))
sm2 = exp2 / np.sum(exp2)
print(f"  Softmax: {sm2}")
print(f"  Live(idx2): {sm2[1]:.4f}")

# Test 3: BGR directly (no RGB conversion)
img3 = cv2.resize(face_bgr, (80, 80))
img3 = img3.astype(np.float32)
img3 = np.transpose(img3, (2, 0, 1))[None, :]

outputs3 = session.run(None, {"input": img3})
print(f"\nTest BGR [0,255]:")
print(f"  Raw: {outputs3[0]}")
scores3 = outputs3[0][0]
exp3 = np.exp(scores3 - np.max(scores3))
sm3 = exp3 / np.sum(exp3)
print(f"  Softmax: {sm3}")
print(f"  Live(idx2): {sm3[2]:.4f}")
