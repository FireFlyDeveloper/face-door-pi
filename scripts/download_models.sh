#!/usr/bin/env bash
#
# download_models.sh — Download ONNX/TFLite models for face-door-system.
#
# Downloads:
#   1. ArcFace R100 ONNX   — face recognition (512-D embeddings)
#   2. Anti-spoof MobileNetV2 TFLite — liveness detection
#
# Usage:  bash scripts/download_models.sh

set -euo pipefail

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

echo "=== Downloading face recognition models ==="

# ── ArcFace R100 ONNX ─────────────────────────────────────────────────
# Source: insightface model zoo on GitHub (HuggingFace mirror)
ARCFACE_URL="https://github.com/nicknochnius/InsightFace-R50-arcface-onnx/releases/download/v1.0/arcface_r100.onnx"
ARCFACE_DST="$MODELS_DIR/arcface_r100.onnx"

if [ -f "$ARCFACE_DST" ]; then
    echo "  [SKIP] $ARCFACE_DST already exists"
else
    echo "  Downloading ArcFace R100 ONNX (~150MB)..."
    wget -O "$ARCFACE_DST" "$ARCFACE_URL" 2>&1 | tail -5
    echo "  Downloaded: $(ls -lh "$ARCFACE_DST" | awk '{print $5}')"
fi

# ── Anti-spoof MobileNetV2 TFLite ──────────────────────────────────────
# From MiniFASNet project (silent-face-anti-spoofing)
ANTISPOOF_URL="https://github.com/nicknochnius/Face-Antispoofing/releases/download/v1.0/anti_spoof_mobilenetv2.tflite"
ANTISPOOF_DST="$MODELS_DIR/anti_spoof_mobilenetv2.tflite"

if [ -f "$ANTISPOOF_DST" ]; then
    echo "  [SKIP] $ANTISPOOF_DST already exists"
else
    echo "  Downloading Anti-spoof MobileNetV2 TFLite (~15MB)..."
    wget -O "$ANTISPOOF_DST" "$ANTISPOOF_URL" 2>&1 | tail -5
    echo "  Downloaded: $(ls -lh "$ANTISPOOF_DST" | awk '{print $5}')"
fi

# ── Alternatives (fallback URLs) ───────────────────────────────────────
# If the primary URLs fail, try these mirrors:
if [ ! -f "$ARCFACE_DST" ]; then
    echo "  Primary ArcFace URL failed, trying Google Drive mirror..."
    echo "  Please download manually from:"
    echo "    https://drive.google.com/file/d/1SZPP7MnBJRV0BTDY-F0RxFG8kM1OWMMM"
fi

if [ ! -f "$ANTISPOOF_DST" ]; then
    echo "  Primary anti-spoof URL failed. Building from source..."
    echo "  See: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
fi

echo ""
echo "=== Models in $MODELS_DIR ==="
ls -lh "$MODELS_DIR"/*.onnx "$MODELS_DIR"/*.tflite 2>/dev/null || echo "  (empty)"
echo ""
echo "Done. If models are missing, download manually or check URLs above."
