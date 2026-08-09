#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

# YOLOv8n ONNX at 320x320.
# The prebuilt ultralytics asset is exported at 640x640, which is ~20x slower on
# the Pi5 CPU (~800ms vs ~40ms/frame). We export at 320 instead — this must match
# detection_input_size. Requires ultralytics (pulls torch); only needed at setup.
YOLO_OUT="$MODELS_DIR/yolov8n.onnx"
if [ ! -f "$YOLO_OUT" ]; then
  echo "Exporting YOLOv8n ONNX at 320x320..."
  python -c "from ultralytics import YOLO; YOLO('yolov8n.pt').export(format='onnx', imgsz=320, simplify=True)"
  mv yolov8n.onnx "$YOLO_OUT"
  rm -f yolov8n.pt
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
