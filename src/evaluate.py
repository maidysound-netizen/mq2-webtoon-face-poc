#!/usr/bin/env python3
"""모델이 찾은 얼굴 박스를 기존 faces.yaml 과 견준다.

기준(faces.yaml)은 모델을 돌리기 전부터 있던 좌표다. AI 오케스트레이션이 그림을 보고 적은 것이라 완전한 정답은 아니다.
얼굴이 아닌 항목(hand, bowl, nametag, body)은 정답에서 뺀다.

판정 기준 (말풍선 피하기 용도에 맞춘 것):
  - 맞힘: 정답 얼굴 넓이의 cover 이상을 모델 박스들이 덮었다
          (말풍선이 피하려면 얼굴 자리가 덮여 있으면 된다. 박스 모양이 꼭 같을 필요는 없다)
  - 놓침: 그만큼 덮지 못했다 → 그 얼굴 위에 말풍선이 올라갈 수 있다
  - 남는 박스: 어떤 정답 얼굴과도 겹치지 않는 박스 → 말풍선 자리를 괜히 막는다
    (정답이 없는 컷의 박스는 '확인 필요'로 따로 센다. 대사가 없어 안 적었을 수 있다)

사용 예:
  python src/evaluate.py --faces <faces.yaml> --model face_detect_v1.4_s
  python src/evaluate.py --faces <faces.yaml> --model face_detect_v1.4_s --threshold 0.1
"""
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
NOT_FACE = {"hand", "bowl", "nametag", "body"}


def coverage(gt, preds, grid=64):
    """정답 박스 넓이 중 모델 박스들이 덮은 비율. 격자로 근사한다."""
    x0, y0, x1, y1 = gt
    xs = np.linspace(x0, x1, grid)
    ys = np.linspace(y0, y1, grid)
    X, Y = np.meshgrid(xs, ys)
    hit = np.zeros_like(X, dtype=bool)
    for p in preds:
        hit |= (X >= p[0]) & (X <= p[2]) & (Y >= p[1]) & (Y <= p[3])
    return float(hit.mean())


def found(gt, preds):
    """박스 크기는 따지지 않고, 그 얼굴을 '찾기는 했는지'만 본다.
    모델 박스 중심이 정답 안에 있거나, 정답 중심이 모델 박스 안에 있으면 찾은 것."""
    gx, gy = (gt[0] + gt[2]) / 2, (gt[1] + gt[3]) / 2
    for p in preds:
        px, py = (p[0] + p[2]) / 2, (p[1] + p[3]) / 2
        if gt[0] <= px <= gt[2] and gt[1] <= py <= gt[3]:
            return True
        if p[0] <= gx <= p[2] and p[1] <= gy <= p[3]:
            return True
    return False


def overlaps(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def evaluate(gt_all, det, thr, cover):
    rows, missed, extra, unlabeled = [], [], [], []
    tp = fn = fp = fd = 0
    for name, d in det.items():
        preds = [b for b in d["boxes"] if b[4] >= thr]
        gts = gt_all.get(name)
        if gts is None:
            if preds:
                unlabeled.append({"panel": name, "boxes": len(preds)})
            continue
        for who, box in gts.items():
            c = coverage(box, preds)
            ok = c >= cover
            f = found(box, preds)
            fd += f
            tp += ok
            fn += not ok
            rows.append({"panel": name, "who": who, "coverage": round(c, 3), "hit": ok, "found": f})
            if not ok:
                missed.append({"panel": name, "who": who, "coverage": round(c, 3), "found": f})
        for p in preds:
            if not any(overlaps(p, g) for g in gts.values()):
                fp += 1
                extra.append({"panel": name, "score": p[4]})
    total = tp + fn
    return {
        "threshold": thr, "cover": cover,
        "faces": total, "hit": tp, "missed": fn,
        "recall": tp / total if total else 0.0,
        "found": fd, "found_rate": fd / total if total else 0.0,
        "extra_boxes": fp,
        "panels_without_labels_but_boxes": len(unlabeled),
    }, rows, missed, extra, unlabeled


def load_gt(path):
    raw = yaml.safe_load(Path(path).read_text())
    gt = {}
    for k, v in raw.items():
        faces = {who: box for who, box in (v or {}).items() if who not in NOT_FACE}
        if faces:
            gt[str(k)] = faces
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--faces", required=True, help="기존 faces.yaml (비교 기준)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--threshold", type=float, default=None, help="박스를 남길 점수. 없으면 여러 값을 훑는다")
    ap.add_argument("--cover", type=float, default=0.5, help="정답 얼굴을 이만큼 덮으면 맞힘")
    a = ap.parse_args()

    res_dir = ROOT / "results" / a.model
    det = json.loads((res_dir / "detections.json").read_text())
    gt = load_gt(a.faces)

    if a.threshold is None:
        sweep = [evaluate(gt, det, t, a.cover)[0] for t in
                 (0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7)]
        (res_dir / "threshold_sweep.json").write_text(json.dumps(sweep, indent=1))
        print(f"{'임계값':>6} {'얼굴':>4} {'맞힘':>4} {'놓침':>4} {'재현율':>6} {'찾음':>4} {'찾음률':>6} {'남는박스':>6} {'정답없는컷에박스':>8}")
        for s in sweep:
            print(f"{s['threshold']:>8} {s['faces']:>5} {s['hit']:>5} {s['missed']:>5} "
                  f"{s['recall']:>8.3f} {s['found']:>5} {s['found_rate']:>8.3f} {s['extra_boxes']:>8} {s['panels_without_labels_but_boxes']:>12}")
        return

    summary, rows, missed, extra, unlabeled = evaluate(gt, det, a.threshold, a.cover)
    out = {"summary": summary, "missed": missed, "extra": extra, "unlabeled_panels": unlabeled, "rows": rows}
    tag = f"thr{a.threshold}"
    (res_dir / f"eval_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print("놓친 얼굴:", [(m["panel"], m["who"], m["coverage"]) for m in missed])
    print("정답 없는 컷에 박스:", [(u["panel"], u["boxes"]) for u in unlabeled])


if __name__ == "__main__":
    main()
