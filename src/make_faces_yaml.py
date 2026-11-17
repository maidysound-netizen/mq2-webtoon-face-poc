#!/usr/bin/env python3
"""모델 박스를 faces.yaml 과 같은 모양으로 적는다. place_bubbles.py 가 그대로 읽을 수 있는 형식.

모델은 누가 누구인지 모르므로 이름 대신 face1, face2 ... 로 적는다.
(말풍선 꼬리를 화자 쪽으로 돌리려면 이름이 필요하다. 그건 다음 단계: 캐릭터 구분)

사용 예:
  python src/make_faces_yaml.py --model head_detect_v2.0_s --threshold 0.1
결과: results/<모델>/faces_auto_thr<임계값>.yaml  (원본 faces.yaml 은 건드리지 않는다)
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, required=True)
    ap.add_argument("--max-faces", type=int, default=6, help="컷당 점수 높은 순으로 이만큼만 적는다")
    a = ap.parse_args()

    res_dir = ROOT / "results" / a.model
    det = json.loads((res_dir / "detections.json").read_text())
    lines = [f"# 자동 생성: {a.model}, 임계값 {a.threshold}, 컷당 최대 {a.max_faces}개",
             "# 컷별 얼굴 위치 [x0, y0, x1, y1] (컷 기준 0~1). 이름은 모른다."]
    for name, d in det.items():
        boxes = sorted([b for b in d["boxes"] if b[4] >= a.threshold], key=lambda b: -b[4])[: a.max_faces]
        if not boxes:
            continue
        parts = [f"face{i + 1}: [{b[0]:.2f}, {b[1]:.2f}, {b[2]:.2f}, {b[3]:.2f}]" for i, b in enumerate(boxes)]
        lines.append(f'"{name}": {{{", ".join(parts)}}}')
    out = res_dir / f"faces_auto_thr{a.threshold}.yaml"
    out.write_text("\n".join(lines) + "\n")
    print("saved", out, f"({len(lines) - 2} panels)")


if __name__ == "__main__":
    main()
