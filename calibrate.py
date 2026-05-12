"""Calibration: capture 8 frames for photo then 8 for live, compute all metrics."""
import cv2, numpy as np, onnxruntime as ort, time, json
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

def lbp_entropy(gray):
    gray = cv2.resize(gray, (64, 64))
    lbp = np.zeros_like(gray, dtype=np.uint8)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    center = gray[1:-1, 1:-1]
    for j,(dy,dx) in enumerate(neighbors):
        ny,nx = 1+dy,1+dx
        n = gray[ny:ny+gray.shape[0]-2,nx:nx+gray.shape[1]-2]
        lbp[1:-1,1:-1] |= ((n >= center).astype(np.uint8)<<j)
    hist,_ = np.histogram(lbp.ravel(), bins=256, range=(0,256))
    hist = hist.astype(np.float32)/(hist.sum()+1e-6)
    return -float(np.sum(hist*np.log2(hist+1e-10)))

def process(frame):
    faces = app.get(frame)
    if not faces:
        return None
    x1,y1,x2,y2 = faces[0].bbox.astype(int)
    crop_bgr = cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_RGB2BGR)
    # LBP
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    ent = lbp_entropy(gray)
    # HSV channels
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    ent_h = lbp_entropy(hsv[:,:,0])
    ent_s = lbp_entropy(hsv[:,:,1])
    ent_v = lbp_entropy(hsv[:,:,2])
    # ONNX
    img = cv2.resize(crop_bgr, (80,80))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = np.transpose(img,(2,0,1))[None,:]
    out = session.run(None,{"input":img})
    raw = out[0][0]
    e = np.exp(raw-np.max(raw))
    sm = e/np.sum(e)
    return {
        "entropy": ent,
        "ent_h": ent_h, "ent_s": ent_s, "ent_v": ent_v,
        "idx0": float(sm[0]), "idx1": float(sm[1]), "idx2": float(sm[2])
    }

print("=== HOLD PHOTO ===")
photo_data = []
for i in range(8):
    picam.capture_array("main")
    time.sleep(0.15)
frame = picam.capture_array("main")
r = process(frame)
if r:
    photo_data.append(r)
    print(f"Photo: ent={r['entropy']:.4f} idx2={r['idx2']:.4f}")

print("=== REMOVE PHOTO (show face) ===")
time.sleep(4)
live_data = []
for i in range(8):
    picam.capture_array("main")
    time.sleep(0.15)
frame = picam.capture_array("main")
r = process(frame)
if r:
    live_data.append(r)
    print(f"Live:  ent={r['entropy']:.4f} idx2={r['idx2']:.4f}")

picam.stop()
picam.close()

if photo_data and live_data:
    p,l = photo_data[0], live_data[0]
    print(f"\n=== CALIBRATION ===")
    print(f"{'Metric':<15} {'Photo':>8} {'Live':>8} {'Diff':>8}")
    print("-"*40)
    for metric in ['entropy','ent_h','ent_s','ent_v','idx0','idx1','idx2']:
        pv, lv = p[metric], l[metric]
        print(f"{metric:<15} {pv:>8.4f} {lv:>8.4f} {pv-lv:>8.4f}")
    
    # Compute best thresholds
    print(f"\n--- Best single-signal thresholds ---")
    
    # ONNX inverted gap
    onnx_p = 1 - p['idx2']
    onnx_l = 1 - l['idx2']
    print(f"ONNX inv gap: photo={onnx_p:.4f} live={onnx_l:.4f} midpoint={(onnx_p+onnx_l)/2:.4f}")
    
    # LBP entropy gap
    ent_p = p['entropy']
    ent_l = l['entropy']
    print(f"LBP ent gap:  photo={ent_p:.4f} live={ent_l:.4f} midpoint={(ent_p+ent_l)/2:.4f}")
    
    # Best combo: find threshold for each LBP_ENTROPY cutoff
    print(f"\n--- Ensemble simulation (0.7*ONNX_inv + 0.3*LBP) ---")
    for thresh in [6.20, 6.25, 6.30, 6.35, 6.40]:
        lbp_p = max(0, min(1, (thresh - ent_p) / 0.30))
        lbp_l = max(0, min(1, (thresh - ent_l) / 0.30))
        combo_p = 0.7*onnx_p + 0.3*lbp_p
        combo_l = 0.7*onnx_l + 0.3*lbp_l
        print(f"  LBP_thr={thresh:.2f}: photo={combo_p:.4f} live={combo_l:.4f} gap={combo_l-combo_p:.4f}")
