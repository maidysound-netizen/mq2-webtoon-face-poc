#!/usr/bin/env bash
# deepghs 의 만화 얼굴·머리 탐지 ONNX 모델을 models/ 아래에 받는다 (MIT).
set -e
cd "$(dirname "$0")"
get() { repo=$1; name=$2; mkdir -p "models/$name"
  for f in model.onnx threshold.json labels.json; do
    curl -sfL -o "models/$name/$f" "https://huggingface.co/deepghs/$repo/resolve/main/$name/$f"; done; echo "ok $name"; }
get anime_face_detection face_detect_v1.4_s
get anime_head_detection head_detect_v2.0_s
