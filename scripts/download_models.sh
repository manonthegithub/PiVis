#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

# YOLOv8n ONNX
YOLO_URL="https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.onnx"
YOLO_OUT="$MODELS_DIR/yolov8n.onnx"
if [ ! -f "$YOLO_OUT" ]; then
  echo "Downloading YOLOv8n ONNX..."
  wget -q --show-progress -O "$YOLO_OUT" "$YOLO_URL"
else
  echo "YOLOv8n already present, skipping."
fi

# Piper voice: en_US-lessac-medium
PIPER_BASE="https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium"
for ext in onnx onnx.json; do
  OUT="$MODELS_DIR/en_US-lessac-medium.$ext"
  if [ ! -f "$OUT" ]; then
    echo "Downloading en_US-lessac-medium.$ext..."
    wget -q --show-progress -O "$OUT" "$PIPER_BASE/en_US-lessac-medium.$ext"
  else
    echo "en_US-lessac-medium.$ext already present, skipping."
  fi
done

echo "Done."
