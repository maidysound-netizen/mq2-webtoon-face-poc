#!/usr/bin/env bash
# 샘플 에피소드 전체 파이프라인: 생성 → 머리 탐지 → 이름 붙이기 → 말풍선 배치 → 세로 웹툰 조립. 전부 로컬.
#
# 준비: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
#       bash download_models.sh
#       .venv/bin/pip install mflux==0.24.0   (1단계 생성을 다시 할 때만. 애플 실리콘 전용, 처음 실행 때 모델 약 16GB를 받음)
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
MODEL=head_detect_v2.0_s
THR=0.1

echo "① 컷 생성 (FLUX.2 klein 4B, 없는 컷만)"
if [ -x .venv/bin/mflux-generate-flux2 ]; then
  $PY src/make_character_sheets.py   # 캐릭터 시트 후보 (확정본이 이미 있으면 컷 생성에 그대로 씀)
  $PY src/generate_samples.py
else
  echo "   mflux 가 없어 생성을 건너뜀 (저장소에 들어 있는 시트와 컷을 씀)"
fi

echo "② 머리 탐지 (YOLO 계열 ONNX, onnxruntime)"
$PY src/detect.py --panels samples/episode/panels --model $MODEL --out results/samples/$MODEL

echo "③ 이름 붙이기 + 조립용 파일 만들기"
$PY src/build_episode.py --model $MODEL --threshold $THR

# ④⑤ 는 작가의 기존 웹툰 스크립트(place_bubbles.py, webtoon.py)로 돌렸다.
#     저작권 문제로 저장소에는 넣지 않았다. 스크립트가 있으면 WEBTOON_DIR 로 폴더를 알려 준다.
#     결과물은 samples/strip_ko.jpg 로 저장소에 들어 있다.
if [ -n "$WEBTOON_DIR" ] && [ -f "$WEBTOON_DIR/place_bubbles.py" ]; then
  echo "④ 말풍선 배치 (얼굴을 피함)"
  $PY "$WEBTOON_DIR/place_bubbles.py" samples/episode
  echo "⑤ 세로 웹툰 조립"
  $PY "$WEBTOON_DIR/webtoon.py" samples/episode --lang ko
  echo "완성: samples/episode/out/ko/strip.png"
else
  echo "④⑤ 말풍선 배치·조립은 작가의 비공개 스크립트라 건너뜀 (결과: samples/strip_ko.jpg)"
fi
