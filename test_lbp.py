"""Quick LBP test - capture, don't wait."""
import cv2, numpy as np, time, sys
from picamera2 import Picamera2

picam = Picamera2()
config = picam.create_video_configuration(main={"size": (640,480), "format":"RGB888"}, controls={"FrameRate":15})
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

gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (64, 64))
lbp = np.zeros_like(gray, dtype=np.uint8)
neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
center = gray[1:-1, 1:-1]
for i, (dy, dx) in enumerate(neighbors):
    ny, nx = 1+dy, 1+dx
    neighbor = gray[ny:ny+gray.shape[0]-2, nx:nx+gray.shape[1]-2]
    lbp[1:-1, 1:-1] |= ((neighbor >= center).astype(np.uint8) << i)

hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0,256))
hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
entropy = -float(np.sum(hist * np.log2(hist + 1e-10)))
score = np.clip((entropy - 6.2) / 2.0, 0.0, 1.0)
print(f"ENTROPY={entropy:.4f} SCORE={score:.4f} VERDICT={'LIVE' if score>=0.5 else 'SPOOF'}")
