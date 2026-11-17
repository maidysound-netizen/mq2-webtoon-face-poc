#!/usr/bin/env python3
"""교차 검증: 오케스트레이션이 적은 faces.yaml 과 로컬 모델 박스가 서로 다른 컷만 골라낸다.

두 방식은 서로 다른 방법으로 얼굴을 찾는다(범용 언어 모델의 추정 vs 전용 탐지 모델).
둘이 같은 실수를 할 가능성은 낮으므로, 결과가 다른 컷만 작가가 보면 된다.

컷을 표시하는 경우:
  A. 기존 좌표의 얼굴을 모델이 못 찾음
     → 기존 좌표가 어긋났거나(오케스트레이션 실수) 모델이 놓쳤다. 둘 중 무엇인지 작가가 본다
  B. 모델이 높은 점수로 찾은 머리가 기존 좌표에 없음
     → 기존 좌표가 얼굴을 빠뜨렸을 수 있다. 대사 없는 엑스트라일 수도 있다

A 는 낮은 임계값(놓치지 않게), B 는 높은 임계값(엑스트라로 표시가 넘치지 않게)을 쓴다.

사용 예:
  python src/crosscheck.py --faces <faces.yaml> --model head_detect_v2.0_s
결과: results/<모델>/crosscheck_a<임계값>_b<임계값>.json
"""
import argparse
import json
from pathlib import Path

from evaluate import found, load_gt, overlaps

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True, help="오케스트레이션이 적은 faces.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--thr-a", type=float, default=0.1, help="A 판정에 쓸 모델 박스 임계값 (낮게)")
    ap.add_argument("--thr-b", type=float, default=0.5, help="B 판정에 쓸 모델 박스 임계값 (높게)")
    a = ap.parse_args()

    res_dir = ROOT / "results" / a.model
    det = json.loads((res_dir / "detections.json").read_text())
    gt = load_gt(a.faces)

    flagged = {}
    for name, d in det.items():
        faces = gt.get(name, {})
        low = [b for b in d["boxes"] if b[4] >= a.thr_a]
        high = [b for b in d["boxes"] if b[4] >= a.thr_b]
        reasons = []
        for who, box in faces.items():
            if not found(box, low):
                reasons.append(f"A: 기존 좌표의 '{who}'를 모델이 못 찾음")
        for b in high:
            if not any(overlaps(b, g) for g in faces.values()):
                reasons.append(f"B: 모델이 찾은 머리(점수 {b[4]:.2f})가 기존 좌표에 없음")
        if reasons:
            flagged[name] = reasons

    total = len(det)
    out = {"model": a.model, "thr_a": a.thr_a, "thr_b": a.thr_b,
           "panels": total, "flagged_panels": len(flagged),
           "review_ratio": len(flagged) / total if total else 0.0, "flagged": flagged}
    path = res_dir / f"crosscheck_a{a.thr_a}_b{a.thr_b}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"확인할 컷: {len(flagged)} / {total} ({out['review_ratio']:.1%})")
    for name, rs in flagged.items():
        print(f"  {name}: " + " / ".join(rs))


if __name__ == "__main__":
    main()
