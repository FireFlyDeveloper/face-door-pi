"""Quick ONNX raw output test - just show your face."""
import cv2, numpy as np, onnxruntime as ort, time, sys
from picamera2 import Picamera2

session = ort.InferenceSession("models/minifasnet_v2.onnx", providers=["CPUExecutionProvider"])

picam = Picamera2()
config = picam.create_video_configuration(
    main={"size": (640,480), "format":"RGB888"},
    controls={"FrameRate":15},
)
picam.configure(config)
picam.start()
time.sleep(2)

for _ in range(10):
    frame = picam.capture_array("main")
    time.sleep(0.1)

picam.stop()
picam.close()

import insightface
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"], root="~/.insightface")
app.prepare(ctx_id=0, det_size=(320, 320))

faces = app.get(frame)
if len(faces) == 0:
    print("NO FACE")
    sys.exit(0)

x1,y1,x2,y2 = faces[0].bbox.astype(int)
face_bgr = cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_RGB2BGR)

# Preprocess (same as anti_spoof.py)
img = cv2.resize(face_bgr, (80, 80))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img.astype(np.float32)  # [0,255]
img = np.transpose(img, (2, 0, 1))[None, :]

outputs = session.run(None, {"input": img})
raw = outputs[0][0]
print(f"Raw: {raw}")
e = np.exp(raw - np.max(raw))
sm = e / np.sum(e)
print(f"Softmax: {sm}")
print(f"Index0={sm[0]:.4f} Index1={sm[1]:.4f} Index2={sm[2]:.4f}")
print(f"ONNX inverted live_score = 1 - idx2 = {1 - sm[2]:.4f}")
