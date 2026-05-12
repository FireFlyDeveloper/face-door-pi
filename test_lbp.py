"""Test LBP on HSV/YCbCr channels for better discrimination."""
import cv2, numpy as np, time, sys
from picamera2 import Picamera2
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"], root="~/.insightface")
app.prepare(ctx_id=0, det_size=(320, 320))

picam = Picamera2()
config = picam.create_video_configuration(main={"size": (640,480), "format":"RGB888"}, controls={"FrameRate":15})
picam.configure(config)
picam.start()
time.sleep(2)

for _ in range(5):
    frame = picam.capture_array("main")
    time.sleep(0.1)

picam.stop()
picam.close()

faces = app.get(frame)
if len(faces) == 0:
    print("NO FACE")
    sys.exit(0)

x1,y1,x2,y2 = faces[0].bbox.astype(int)
face_rgb = frame[y1:y2,x1:x2]  # already RGB
print(f"Face crop: {x2-x1}x{y2-y1}")

def lbp_entropy(gray, resize=64):
    gray = cv2.resize(gray, (resize, resize))
    lbp = np.zeros_like(gray, dtype=np.uint8)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    center = gray[1:-1, 1:-1]
    for j, (dy, dx) in enumerate(neighbors):
        ny, nx = 1+dy, 1+dx
        neighbor = gray[ny:ny+gray.shape[0]-2, nx:nx+gray.shape[1]-2]
        lbp[1:-1, 1:-1] |= ((neighbor >= center).astype(np.uint8) << j)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0,256))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
    return -float(np.sum(hist * np.log2(hist + 1e-10)))

# Gray (original)
gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
print(f"GRAY:      entropy={lbp_entropy(gray):.4f}")

# HSV channels
hsv = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2HSV)
h, s, v = cv2.split(hsv)
print(f"HSV-H:     entropy={lbp_entropy(h):.4f}")
print(f"HSV-S:     entropy={lbp_entropy(s):.4f}")
print(f"HSV-V:     entropy={lbp_entropy(v):.4f}")

# YCbCr channels
ycrcb = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2YCrCb)
y, cr, cb = cv2.split(ycrcb)
print(f"YCrCb-Y:   entropy={lbp_entropy(y):.4f}")
print(f"YCrCb-Cr:  entropy={lbp_entropy(cr):.4f}")
print(f"YCrCb-Cb:  entropy={lbp_entropy(cb):.4f}")

# Gray at 128x128
print(f"GRAY128:   entropy={lbp_entropy(gray, 128):.4f}")

# Gray at 32x32
print(f"GRAY32:    entropy={lbp_entropy(gray, 32):.4f}")

# RGB channels
r, g, b = cv2.split(face_rgb)
print(f"RGB-R:     entropy={lbp_entropy(r):.4f}")
print(f"RGB-G:     entropy={lbp_entropy(g):.4f}")
print(f"RGB-B:     entropy={lbp_entropy(b):.4f}")

# Variance of Laplacian (focus measure)
lap = cv2.Laplacian(gray, cv2.CV_64F)
print(f"LAP_VAR:   {lap.var():.2f}")
