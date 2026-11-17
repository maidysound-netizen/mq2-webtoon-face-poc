#!/usr/bin/env python3
"""웹툰 컷에서 얼굴(또는 머리)을 찾아 박스를 저장한다.

모델은 deepghs 의 YOLO 계열 ONNX 파일을 onnxruntime 으로 돌린다.
ultralytics 라이브러리(AGPL)를 쓰지 않으므로 라이선스 조건이 가볍다.
그림은 이 컴퓨터 밖으로 나가지 않는다.

사용 예:
  python src/detect.py --panels <컷폴더> --model face_detect_v1.4_s
  python src/detect.py --panels <컷폴더> --model head_detect_v2.0_s --min-score 0.01

결과: results/<모델>/detections.json
  {컷이름: {"size": [w, h], "boxes": [[x0, y0, x1, y1, score], ...]}}
  좌표는 컷 기준 0~1. 점수는 반올림하지 않는다(임계값을 나중에 정하려고).
  min-score 는 일부러 낮게 둔다. 임계값은 evaluate.py 에서 바꿔 가며 본다.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def letterbox(img: Image.Image, size: int):
    """비율을 유지한 채 size×size 안에 넣고, 남는 곳은 회색으로 채운다."""
    w, h = img.size
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas.paste(img.resize((nw, nh), Image.BILINEAR), (px, py))
    arr = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
    return arr, r, px, py


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        xx0 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy0 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx1 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy1 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx1 - xx0, 0, None) * np.clip(yy1 - yy0, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
        iou = inter / (area(boxes[i]) + area(boxes[order[1:]]) - inter + 1e-9)
        order = order[1:][iou < iou_thr]
    return keep


def detect(sess, img: Image.Image, size: int, min_score: float, iou_thr: float):
    x, r, px, py = letterbox(img, size)
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    # 모델마다 축 순서가 다르다: (5, anchors) 또는 (anchors, 5). 짧은 쪽이 값 축이다.
    if out.shape[0] > out.shape[1]:
        out = out.T
    cx, cy, bw, bh, score = out[0], out[1], out[2], out[3], out[4:].max(axis=0)
    m = score >= min_score
    cx, cy, bw, bh, score = cx[m], cy[m], bw[m], bh[m], score[m]
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
    keep = nms(boxes, score, iou_thr) if len(score) else []
    w, h = img.size
    res = []
    for i in keep:
        x0, y0, x1, y1 = boxes[i]
        x0, x1 = (x0 - px) / r / w, (x1 - px) / r / w
        y0, y1 = (y0 - py) / r / h, (y1 - py) / r / h
        res.append([float(np.clip(x0, 0, 1)), float(np.clip(y0, 0, 1)),
                    float(np.clip(x1, 0, 1)), float(np.clip(y1, 0, 1)), float(score[i])])
    return res


def panel_files(folder: Path):
    """미리보기(.prev)·시험(.test) 파일과 숨김 파일은 뺀다."""
    for p in sorted(folder.glob("*.png")):
        if p.name.startswith(".") or ".prev" in p.name or ".test" in p.name:
            continue
        yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panels", required=True, help="말풍선 없는 원본 컷 폴더")
    ap.add_argument("--model", required=True, help="models/ 아래 폴더 이름")
    ap.add_argument("--size", type=int, default=640, help="모델 입력 크기")
    ap.add_argument("--min-score", type=float, default=0.01, help="저장할 최소 점수. 낮게 둔다")
    ap.add_argument("--iou", type=float, default=0.5, help="겹친 박스를 하나로 합칠 기준")
    ap.add_argument("--out", default=None, help="결과 폴더. 기본 results/<모델>")
    a = ap.parse_args()

    model_dir = ROOT / "models" / a.model
    sess = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    out_dir = Path(a.out) if a.out else ROOT / "results" / a.model
    out_dir.mkdir(parents=True, exist_ok=True)

    results, times = {}, []
    for p in panel_files(Path(a.panels)):
        img = Image.open(p).convert("RGB")
        t = time.perf_counter()
        boxes = detect(sess, img, a.size, a.min_score, a.iou)
        times.append(time.perf_counter() - t)
        results[p.stem] = {"size": list(img.size), "boxes": boxes}

    (out_dir / "detections.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    meta = {"model": a.model, "panels": len(results), "size": a.size, "min_score": a.min_score,
            "iou": a.iou, "sec_total": sum(times), "sec_per_panel": sum(times) / max(len(times), 1)}
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == "__main__":
    main()
