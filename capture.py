"""Capture single frame, compute all anti-spoof metrics."""
import cv2, numpy as np, onnxruntime as ort, time, sys, json
from picamera2 import Picamera2
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"], root="~/.insightface")
app.prepare(ctx_id=0, det_size=(320, 320))
session = ort.InferenceSession("models/minifasnet_v2.onnx", providers=["CPUExecutionProvider"])

picam = Picamera2()
config = picam.create_video_configuration(main={"size":(640,480),"format":"RGB888"}, controls={"FrameRate":15})
picam.configure(config)
picam.start()
time.sleep(2)

for _ in range(10):
    picam.capture_array("main")
    time.sleep(0.1)
frame = picam.capture_array("main")
picam.stop()
picam.close()

faces = app.get(frame)
if not faces:
    print("NO_FACE")
    sys.exit(0)

x1,y1,x2,y2 = faces[0].bbox.astype(int)
crop_bgr = cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

# LBP entropy
g64 = cv2.resize(gray, (64,64))
lbp = np.zeros_like(g64, dtype=np.uint8)
neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
center = g64[1:-1, 1:-1]
for j,(dy,dx) in enumerate(neighbors):
    ny,nx = 1+dy,1+dx
    n = g64[ny:ny+g64.shape[0]-2,nx:nx+g64.shape[1]-2]
    lbp[1:-1,1:-1] |= ((n >= center).astype(np.uint8)<<j)
hist,_ = np.histogram(lbp.ravel(), bins=256, range=(0,256))
hist = hist.astype(np.float32)/(hist.sum()+1e-6)
entropy = -float(np.sum(hist*np.log2(hist+1e-10)))

# HSV channels
hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
def single_lbp_entropy(ch):
    g = cv2.resize(ch, (64,64))
    l = np.zeros_like(g, dtype=np.uint8)
    c = g[1:-1,1:-1]
    for j,(dy,dx) in enumerate(neighbors):
        ny,nx = 1+dy,1+dx
        n = g[ny:ny+g.shape[0]-2,nx:nx+g.shape[1]-2]
        l[1:-1,1:-1] |= ((n >= c).astype(np.uint8)<<j)
    h,_ = np.histogram(l.ravel(), bins=256, range=(0,256))
    h = h.astype(np.float32)/(h.sum()+1e-6)
    return -float(np.sum(h*np.log2(h+1e-10)))
ent_h = single_lbp_entropy(hsv[:,:,0])
ent_s = single_lbp_entropy(hsv[:,:,1])
ent_v = single_lbp_entropy(hsv[:,:,2])

# ONNX
img = cv2.resize(crop_bgr, (80,80))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
img = np.transpose(img,(2,0,1))[None,:]
out = session.run(None,{"input":img})
raw = out[0][0]
e = np.exp(raw-np.max(raw))
sm = e/np.sum(e)

print(f"ENTROPY={entropy:.4f}")
print(f"ENT_H={ent_h:.4f}")
print(f"ENT_S={ent_s:.4f}")
print(f"ENT_V={ent_v:.4f}")
print(f"ONNX_IDX0={sm[0]:.4f}")
print(f"ONNX_IDX1={sm[1]:.4f}")
print(f"ONNX_IDX2={sm[2]:.4f}")
print(f"ONNX_INV={1-sm[2]:.4f}")
