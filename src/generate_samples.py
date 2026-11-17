#!/usr/bin/env python3
"""FLUX.2 klein 4B(mflux, 로컬)로 samples/story.yaml 의 컷을 그린다.

실제 원고 대신 이 샘플을 저장소에 넣어, 누구나 clone 한 뒤 전체 파이프라인을 돌려 볼 수 있게 한다.
발행용 그림이 아니라 시연·시험용 데이터다. 품질은 실제 작업에 쓰는 생성 서비스에 못 미친다.

캐릭터 일관성:
  인물이 나오는 컷은 확정된 캐릭터 시트(samples/characters/<이름>.png)를 참고 그림으로 넣는다
  (mflux-generate-flux2-edit 의 여러 장 참고). 시트가 없으면 글 설명만으로 그린다.
  캐릭터 시트는 src/make_character_sheets.py 로 먼저 만든다.

준비:
  .venv/bin/pip install mflux==0.24.0     # MIT, 애플 실리콘 전용
  (처음 실행할 때 black-forest-labs/FLUX.2-klein-4B 를 받는다. 약 16GB, Apache 2.0)

사용 예:
  python src/generate_samples.py                  # 없는 컷만 생성
  python src/generate_samples.py --only 03 07     # 몇 컷만
  python src/generate_samples.py --only 03 --force --seed 7   # 다시 뽑기
결과: samples/episode/panels/<id>.png, samples/generation_log.json (프롬프트·참고 그림·시드·걸린 시간)
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
BIN = Path(sys.executable).parent
MODEL = "flux2-klein-4b"
SIZES = {"tall": (768, 1024), "wide": (1024, 640), "square": (768, 768)}
SHEETS = ROOT / "samples" / "characters"


def build_prompt(story, panel, refs):
    scene = panel["scene"]
    for k, v in story["characters"].items():
        scene = scene.replace("{" + k + "}", v)
    head = ""
    if refs:
        who = "; ".join(f"image {i} shows {story['characters'][n]}" for i, n in enumerate(refs, 1))
        head = (f"Reference character sheets: {who}. Draw these same characters with the same face, "
                f"hair and clothes, as a single comic panel, not a character sheet. ")
    return f"{head}{story['style']}, {scene}, no text, no speech bubbles"


def generate(prompt, refs, seed, w, h, steps, out):
    if refs:
        cmd = [str(BIN / "mflux-generate-flux2-edit"), "--image-paths", *[str(SHEETS / f"{n}.png") for n in refs]]
    else:
        cmd = [str(BIN / "mflux-generate-flux2")]
    cmd += ["--model", MODEL, "--prompt", prompt, "--seed", str(seed), "--steps", str(steps),
            "--width", str(w), "--height", str(h), "--output", str(out)]
    subprocess.run(cmd, check=True)
    alt = out.with_name(out.stem + "_1.png")  # mflux 는 이름 끝에 _1 을 붙여 저장한다
    if alt.exists():
        alt.replace(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--story", default=str(ROOT / "samples" / "story.yaml"))
    ap.add_argument("--out", default=str(ROOT / "samples" / "episode" / "panels"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seed", type=int, default=100, help="컷마다 seed+번호 를 쓴다")
    ap.add_argument("--steps", type=int, default=4, help="klein 은 적은 단계로 그리도록 만든 모델")
    ap.add_argument("--no-refs", action="store_true", help="캐릭터 시트를 쓰지 않고 글 설명만으로 그린다")
    ap.add_argument("--force", action="store_true", help="이미 있는 파일도 다시 만든다")
    a = ap.parse_args()

    story = yaml.safe_load(Path(a.story).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out.parent.parent / "generation_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    for i, p in enumerate(story["panels"]):
        pid = p["id"]
        if a.only and pid not in a.only:
            continue
        path = out / f"{pid}.png"
        if path.exists() and not a.force:
            continue
        w, h = SIZES[p["size"]]
        cast = p.get("cast") or []
        refs = [] if a.no_refs else [n for n in cast if (SHEETS / f"{n}.png").exists()]
        prompt = build_prompt(story, p, refs)
        seed = a.seed + i
        t = time.perf_counter()
        generate(prompt, refs, seed, w, h, a.steps, path)
        sec = time.perf_counter() - t
        log[pid] = {"prompt": prompt, "refs": refs, "seed": seed, "size": [w, h], "steps": a.steps,
                    "model": MODEL, "sec": round(sec, 1)}
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
        print(f"{pid}: {sec:.1f}초 (참고 {refs or '없음'})", flush=True)


if __name__ == "__main__":
    main()
