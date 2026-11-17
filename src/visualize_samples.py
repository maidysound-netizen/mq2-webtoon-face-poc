#!/usr/bin/env python3
"""샘플 에피소드 컷에 머리 탐지 박스와 붙인 이름을 그린다 (결과 페이지 docs/ 용).

빨강: 대본의 인물 이름이 붙은 머리 / 주황: 이름 없는 엑스트라 / 회색: 임계값 미만
사용 예:
  python src/visualize_samples.py --threshold 0.1
결과: docs/img/sample/vis/<컷>.jpg, docs/img/sample/panels/<컷>.jpg, docs/sample_data.json
"""
import argparse
import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
FONT = "/System/Library/Fonts/AppleSDGothicNeo.ttc"
NAMES = {"facil": "퍼실", "hana": "하나", "jun": "준", "corgi": "코기"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="head_detect_v2.0_s")
    ap.add_argument("--threshold", type=float, default=0.1)
    a = ap.parse_args()

    story = yaml.safe_load((ROOT / "samples" / "story.yaml").read_text())
    det = json.loads((ROOT / "results" / "samples" / a.model / "detections.json").read_text())
    faces = yaml.safe_load((ROOT / "samples" / "episode" / "faces.yaml").read_text()) or {}
    out_v, out_p = ROOT / "docs" / "img" / "sample" / "vis", ROOT / "docs" / "img" / "sample" / "panels"
    out_v.mkdir(parents=True, exist_ok=True)
    out_p.mkdir(parents=True, exist_ok=True)
    data = []
    for p in story["panels"]:
        pid = p["id"]
        im = Image.open(ROOT / "samples" / "episode" / "panels" / f"{pid}.png").convert("RGB")
        im.thumbnail((720, 1000))
        im.save(out_p / f"{pid}.jpg", quality=82, optimize=True)
        W, H = im.size
        d = ImageDraw.Draw(im)
        f = ImageFont.truetype(FONT, max(14, W // 40))
        named = {tuple(v): k for k, v in (faces.get(pid) or {}).items()}
        n_main = n_extra = 0
        for b in sorted(det.get(pid, {}).get("boxes", []), key=lambda b: b[4]):
            x0, y0, x1, y1, s = b
            r = [x0 * W, y0 * H, x1 * W, y1 * H]
            if s < a.threshold:
                d.rectangle(r, outline=(160, 160, 160), width=1)
                continue
            who = named.get(tuple(round(v, 3) for v in b[:4]), "")
            extra = who.startswith("extra") or not who
            n_extra += extra
            n_main += not extra
            col = (245, 158, 11) if extra else (220, 38, 38)
            d.rectangle(r, outline=col, width=3)
            label = f"{'엑스트라' if extra else NAMES.get(who, who)} {s:.2f}"
            tw = d.textlength(label, font=f)
            ty = max(0, r[1] - f.size - 6)
            d.rectangle([r[0], ty, r[0] + tw + 8, ty + f.size + 6], fill=col)
            d.text((r[0] + 4, ty + 2), label, fill="white", font=f)
        im.save(out_v / f"{pid}.jpg", quality=82, optimize=True)
        data.append({"id": pid, "cast": [NAMES.get(c, c) for c in p.get("cast") or []],
                     "named": n_main, "extra": n_extra,
                     "lines": [ln["ko"].replace("\n", " ") for ln in p.get("lines") or []]})
    (ROOT / "docs" / "sample_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1))
    print(f"{len(data)}컷, 이름 붙은 머리 {sum(x['named'] for x in data)}, 엑스트라 {sum(x['extra'] for x in data)}")


if __name__ == "__main__":
    main()
