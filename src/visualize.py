#!/usr/bin/env python3
"""기존 좌표(faces.yaml)와 모델 박스를 컷 위에 함께 그린다. 숫자만 보지 않고 눈으로 검산하려는 것.

  초록 실선  : 기존 좌표 (faces.yaml)
  빨강       : 모델 박스 중 임계값을 넘은 것 (점수 표시)
  회색 점선  : 임계값을 못 넘은 박스 (참고용)

사용 예:
  python src/visualize.py --panels <컷폴더> --faces <faces.yaml> --model head_detect_v2.0_s --threshold 0.1
결과: results/<모델>/vis_thr<임계값>/<컷>.jpg  (덮어쓰지 않도록 임계값별 폴더)
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from evaluate import load_gt

ROOT = Path(__file__).resolve().parent.parent


def px(box, w, h):
    return [box[0] * w, box[1] * h, box[2] * w, box[3] * h]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", required=True)
    ap.add_argument("--faces", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--only", nargs="*", help="이 컷들만 그린다")
    ap.add_argument("--width", type=int, default=640, help="저장할 그림 폭")
    a = ap.parse_args()

    res_dir = ROOT / "results" / a.model
    det = json.loads((res_dir / "detections.json").read_text())
    gt = load_gt(a.faces)
    out = res_dir / f"vis_thr{a.threshold}"
    out.mkdir(parents=True, exist_ok=True)

    for name, d in det.items():
        if a.only and name not in a.only:
            continue
        img = Image.open(Path(a.panels) / f"{name}.png").convert("RGB")
        w, h = img.size
        dr = ImageDraw.Draw(img)
        lw = max(3, w // 200)
        for b in d["boxes"]:
            if b[4] < a.threshold:
                dr.rectangle(px(b, w, h), outline=(160, 160, 160), width=max(1, lw // 2))
        for who, b in gt.get(name, {}).items():
            dr.rectangle(px(b, w, h), outline=(0, 220, 0), width=lw)
            dr.text((b[0] * w + 6, b[1] * h + 4), who, fill=(0, 255, 0))
        for b in d["boxes"]:
            if b[4] >= a.threshold:
                dr.rectangle(px(b, w, h), outline=(255, 40, 40), width=lw)
                dr.text((b[0] * w + 6, b[3] * h - 16), f"{b[4]:.3f}", fill=(255, 60, 60))
        img.thumbnail((a.width, a.width * 4))
        img.save(out / f"{name}.jpg", quality=85)
    print("saved to", out)


if __name__ == "__main__":
    main()
