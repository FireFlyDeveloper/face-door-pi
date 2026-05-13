"""Calibration: capture 1 photo frame then 1 live frame, compute all metrics.
Interactive — press Enter after positioning each target."""

import cv2, numpy as np, onnxruntime as ort, time, json, sys
from picamera2 import Picamera2
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(name="buffalo_s", allowed_modules=["detection"], root="~/.insightface")
app.prepare(ctx_id=0, det_size=(320, 320))

session = ort.InferenceSession("models/minifasnet_v2.onnx", providers=["CPUExecutionProvider"])

picam = Picamera2()
config = picam.create_video_configuration(
    main={"size": (640, 480), "format": "RGB888"},
    controls={"FrameRate": 15},
)
picam.configure(config)
picam.start()
time.sleep(2)


def lbp_entropy(gray):
    gray = cv2.resize(gray, (64, 64))
    lbp = np.zeros_like(gray, dtype=np.uint8)
    neighbors = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    center = gray[1:-1, 1:-1]
    for j, (dy, dx) in enumerate(neighbors):
        ny, nx = 1 + dy, 1 + dx
        n = gray[ny:ny+gray.shape[0]-2, nx:nx+gray.shape[1]-2]
        lbp[1:-1, 1:-1] |= ((n >= center).astype(np.uint8) << j)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype(np.float32) / (hist.sum() + 1e-6)
    return -float(np.sum(hist * np.log2(hist + 1e-10)))


def process(frame):
    faces = app.get(frame)
    if not faces:
        return None
    x1, y1, x2, y2 = faces[0].bbox.astype(int)
    crop_bgr = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_RGB2BGR)
    # LBP
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    ent = lbp_entropy(gray)
    # HSV channels
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    ent_h = lbp_entropy(hsv[:, :, 0])
    ent_s = lbp_entropy(hsv[:, :, 1])
    ent_v = lbp_entropy(hsv[:, :, 2])
    # ONNX MiniFASNet
    img = cv2.resize(crop_bgr, (80, 80))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))[None, :]
    out = session.run(None, {"input": img})
    raw = out[0][0]
    e = np.exp(raw - np.max(raw))
    sm = e / np.sum(e)
    return {
        "entropy": ent,
        "ent_h": ent_h, "ent_s": ent_s, "ent_v": ent_v,
        "idx0": float(sm[0]), "idx1": float(sm[1]), "idx2": float(sm[2]),
    }


def wait_and_capture(label):
    """Wait for Enter, then warm up and capture one frame."""
    input(f"\n>>> {label}\n    Position it in front of the camera, then press Enter: ")
    print("    Warming up (8 frames)...")
    for i in range(8):
        picam.capture_array("main")
        time.sleep(0.15)
    print("    Capturing...")
    frame = picam.capture_array("main")
    r = process(frame)
    if r:
        print(f"    OK — ent={r['entropy']:.4f}  idx2={r['idx2']:.4f}")
    else:
        print("    No face detected — try again later.")
    return r


# ── Interactive calibration ─────────────────────────────────────────────

print("\n" + "=" * 50)
print("  FACE-DOOR ANTI-SPOOF CALIBRATOR")
print("=" * 50)

photo = wait_and_capture("STEP 1: Hold a PRINTED PHOTO")
live = wait_and_capture("STEP 2: Show your LIVE face (no photo)")

picam.stop()
picam.close()

# ── Results ─────────────────────────────────────────────────────────────

if photo is None or live is None:
    print("\n❌ Calibration incomplete. One or both captures had no face detected.")
    sys.exit(1)

p, l = photo, live
print(f"\n{'='*50}")
print(f"  CALIBRATION RESULTS")
print(f"{'='*50}")
print(f"{'Metric':<15} {'Photo':>10} {'Live':>10} {'Diff':>10}")
print(f"{'-'*45}")
for metric in ["entropy", "ent_h", "ent_s", "ent_v", "idx0", "idx1", "idx2"]:
    pv, lv = p[metric], l[metric]
    diff = pv - lv
    arrow = "⬆ photo higher" if diff > 0 else "⬇ live higher"
    print(f"{metric:<15} {pv:>10.4f} {lv:>10.4f} {diff:>+10.4f}  {arrow}")

print(f"\n{'─'*45}")
print(f"  BEST THRESHOLDS")
print(f"{'─'*45}")

# ONNX inverted (1 - idx2) — higher = more live-like
onnx_p = 1 - p["idx2"]
onnx_l = 1 - l["idx2"]
onnx_mid = (onnx_p + onnx_l) / 2
print(f"  ONNX inv gap:    photo={onnx_p:.4f}  live={onnx_l:.4f}")
print(f"  Recommended ONNX threshold: {onnx_mid:.4f}  (midpoint)")

# LBP entropy — higher = more live-like
ent_p = p["entropy"]
ent_l = l["entropy"]
ent_mid = (ent_p + ent_l) / 2
print(f"  LBP ent gap:     photo={ent_p:.4f}  live={ent_l:.4f}")
print(f"  Recommended LBP threshold: {ent_mid:.4f}  (midpoint)")

# Ensemble simulation
print(f"\n{'─'*45}")
print(f"  ENSEMBLE SIMULATION (60% ONNX_inv + 40% LBP)")
print(f"{'─'*45}")
for thresh in [5.80, 5.90, 6.00, 6.10, 6.20, 6.30]:
    lbp_p_norm = max(0, min(1, (thresh - ent_p) / 0.30))
    lbp_l_norm = max(0, min(1, (thresh - ent_l) / 0.30))
    combo_p = 0.6 * onnx_p + 0.4 * lbp_p_norm
    combo_l = 0.6 * onnx_l + 0.4 * lbp_l_norm
    gap = combo_l - combo_p
    status = "✅" if gap > 0.05 else "⚠️" if gap > 0 else "❌"
    print(f"  LBP thr={thresh:.2f}:  photo={combo_p:.4f}  live={combo_l:.4f}  gap={gap:+.4f}  {status}")

print()
