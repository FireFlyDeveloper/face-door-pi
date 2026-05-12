#!/usr/bin/env bash
#
# download_models.sh — Download face recognition models for face-door-system.
#
# Downloads:
#   1. MiniFASNet v2 ONNX — anti-spoof liveness detection (from silent-anti-spoof)
#   2. (Optional) insightface buffalo models for ArcFace-style recognition
#
# Usage:  bash scripts/download_models.sh

set -euo pipefail

MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"

echo "=== Checking models directory ==="
ls -lh "$MODELS_DIR/" 2>/dev/null || echo "  (empty)"

echo ""
echo "=== Anti-spoof: MiniFASNet ONNX ==="
MINI_MODEL="$MODELS_DIR/minifasnet_v2.onnx"
if [ -f "$MINI_MODEL" ] && [ -s "$MINI_MODEL" ]; then
    echo "  [OK] $MINI_MODEL ($(stat -c%s "$MINI_MODEL" 2>/dev/null || stat -f%z "$MINI_MODEL" 2>/dev/null) bytes)"
else
    echo "  [MISSING] Place minifasnet_v2.onnx in $MODELS_DIR/"
    echo "  Source: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing"
    echo "  Or copy from the existing project at silent-anti-spoof/resources/"
fi

echo ""
echo "=== Status ==="
ls -lh "$MODELS_DIR/" 2>/dev/null || echo "  (no models)"
echo ""
echo "Done. At minimum, minifasnet_v2.onnx is needed for anti-spoof."
