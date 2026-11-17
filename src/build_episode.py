#!/usr/bin/env python3
"""탐지 결과에 이름을 붙여 faces.yaml 을 만들고, 조립기가 읽는 episode.yaml·episode_dialogue.yaml 을 쓴다.

이름 붙이기 규칙(단순):
  - 컷의 cast 는 화면 왼쪽부터 적혀 있다(story.yaml).
  - 임계값을 넘은 머리 박스 중 점수 높은 것을 cast 수만큼 고르고, x 중심 순으로 정렬해 이름을 붙인다.
  - 남는 박스는 extra1, extra2 … 로 남긴다. 말풍선은 이 얼굴도 피한다.
  - 박스가 cast 보다 적으면 그 컷을 '확인 필요'로 기록한다(예: 뒷모습).

사용 예:
  python src/build_episode.py --model head_detect_v2.0_s --threshold 0.1
결과: samples/episode/{faces.yaml, episode.yaml, episode_dialogue.yaml}, samples/naming_report.json
"""
import argparse
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"
EP = SAMPLES / "episode"


def name_faces(story, det, thr):
    faces, report = {}, {}
    for p in story["panels"]:
        pid, cast = p["id"], p.get("cast") or []
        boxes = sorted([b for b in det.get(pid, {}).get("boxes", []) if b[4] >= thr], key=lambda b: -b[4])
        main, extra = boxes[: len(cast)], boxes[len(cast):]
        main.sort(key=lambda b: (b[0] + b[2]) / 2)
        named = {who: [round(v, 3) for v in b[:4]] for who, b in zip(cast, main)}
        for i, b in enumerate(extra, 1):
            named[f"extra{i}"] = [round(v, 3) for v in b[:4]]
        if named:
            faces[pid] = named
        missing = cast[len(main):]
        if missing or (not cast and boxes):
            report[pid] = {"cast": cast, "boxes": len(boxes), "missing": missing,
                           "note": "얼굴이 cast 보다 적음" if missing else "인물 없는 컷에서 박스가 나옴"}
    return faces, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="head_detect_v2.0_s")
    ap.add_argument("--threshold", type=float, default=0.1)
    a = ap.parse_args()

    story = yaml.safe_load((SAMPLES / "story.yaml").read_text())
    det = json.loads((ROOT / "results" / "samples" / a.model / "detections.json").read_text())

    faces, report = name_faces(story, det, a.threshold)
    (EP / "faces.yaml").write_text(
        f"# 자동 생성: {a.model}, 임계값 {a.threshold}. 이름은 story.yaml cast 순서(왼쪽부터)로 붙임\n"
        + yaml.safe_dump(faces, allow_unicode=True, sort_keys=False, default_flow_style=None, width=200))
    (SAMPLES / "naming_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))

    dialogue = {}
    for p in story["panels"]:
        cast = p.get("cast") or []
        lines = []
        for ln in p.get("lines") or []:
            who = ln.get("who", "sfx" if ln["type"] == "sfx" else "narration")
            side = "center"
            if who in cast and len(cast) > 1:
                i = cast.index(who)
                mid = (len(cast) - 1) / 2
                side = "left" if i < mid else ("right" if i > mid else "center")
            lines.append({"who": who, "type": ln["type"], "side": side, "ko": ln["ko"]})
        dialogue[p["id"]] = lines
    (EP / "episode_dialogue.yaml").write_text(yaml.safe_dump({"panels": dialogue}, allow_unicode=True, sort_keys=False))

    spec = {"title": {"ko": "퍼실 선생님의 PoC 교실"}, "signature": "mq2-webtoon-face-poc",
            "width": 800, "gap": 40, "langs": ["ko"],
            "panels": [{"image": f"panels/{p['id']}.png", "bubbles": [], "faces": []} for p in story["panels"]]}
    (EP / "episode.yaml").write_text(yaml.safe_dump(spec, allow_unicode=True, sort_keys=False, width=200))
    print(f"faces: {sum(len(v) for v in faces.values())}개 / 확인 필요 컷: {len(report)}")
    for pid, r in report.items():
        print(f"  {pid}: {r['note']} (cast {r['cast']}, 박스 {r['boxes']})")


if __name__ == "__main__":
    main()
