#!/usr/bin/env bash
# Downloads the YOLOX-nano ONNX model (Apache-2.0, from the official YOLOX release) for local runs.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
URL="https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
curl -fL --retry 3 -o models/yolox_nano.onnx "$URL"
ls -lh models/yolox_nano.onnx
