"""Test MiniFASNet ONNX outputs with a real photo face."""
import cv2
import numpy as np
import onnxruntime as ort
import logging
import sys
import time

logging.basicConfig(level=logging.DEBUG)

# Capture via picamera2
from picamera2 import Picamera2

picam = Picamera2()
config = picam.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 15},
)
picam.configure(config)
picam.start()

time.sleep(2)  # let AEC/AGC settle

for i in range(3):
    frame = picam.capture_array("main")
    time.sleep(0.5)

picam.stop()
picam.close()

print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}, range: [{frame.min()}, {frame.max()}]")

# Detect faces
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
face_crop = frame[y1:y2, x1:x2]
print(f"Face crop: {face_crop.shape}")

# MiniFASNet expects BGR input (camera gives RGB, convert to BGR)
face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)

# Load model
session = ort.InferenceSession("models/minifasnet_v2.onnx", providers=["CPUExecutionProvider"])

# Preprocess
img = cv2.resize(face_bgr, (80, 80))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # MiniFASNet expects RGB
img = img.astype(np.float32) / 255.0
img = np.transpose(img, (2, 0, 1))[None, :]

outputs = session.run(None, {"input": img})
print(f"Raw output ({outputs[0].shape}): {outputs[0]}")

scores = outputs[0][0]
exp = np.exp(scores - np.max(scores))
sm = exp / np.sum(exp)
print(f"Softmax: {sm}")
print(f"Live (idx 2): {sm[2]:.4f}")

# Try alternative interpretations
print(f"\nSigmoid: {1.0 / (1.0 + np.exp(-scores))}")
print(f"Direct raw: {scores}")
