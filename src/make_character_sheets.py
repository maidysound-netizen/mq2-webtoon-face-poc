#!/usr/bin/env python3
"""컷을 그리기 전에 캐릭터 시트(설정화)를 먼저 만든다. FLUX.2 klein 4B, mflux, 전부 로컬.

캐릭터 설명은 samples/story.yaml 의 characters 를 그대로 쓴다.
캐릭터마다 시드를 몇 개 뽑아 두고, 사람이 하나를 고른다(samples/characters/<이름>.png 로 복사).
고른 시트는 컷 생성 때 참고 그림으로 들어가서, 컷마다 얼굴과 옷이 같게 나오도록 돕는다.

준비:
  .venv/bin/pip install mflux==0.24.0     # MIT, 애플 실리콘 전용
  (처음 실행할 때 Hugging Face 에서 black-forest-labs/FLUX.2-klein-4B 를 받는다. 약 16GB, Apache 2.0)

사용 예:
  python src/make_character_sheets.py                    # 캐릭터마다 시드 3개
  python src/make_character_sheets.py --only corgi --seeds 7 8 9
  python src/make_character_sheets.py --pick facil 2     # 후보 2번을 확정 → samples/characters/facil.png
결과: samples/characters/candidates/<이름>_s<시드>.png, samples/characters/sheets_log.json
"""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples" / "characters"
CAND = OUT / "candidates"
MFLUX = Path(sys.executable).parent / "mflux-generate-flux2"
MODEL = "flux2-klein-4b"
SIZE = (1280, 768)

SHEET = ("character design reference sheet of {desc}. "
         "Left side: the same character standing full body in front view, side view and back view. "
         "Right side: three head close-ups of the same character, happy, surprised and thinking expressions. "
         "plain white background, korean webtoon style, clean anime line art, flat cel shading, "
         "consistent design across all views, no text, no labels")
SHEET_ANIMAL = ("character design reference sheet of {desc}. "
                "Left side: the same dog full body in front view and side view. "
                "Right side: three head close-ups of the same dog, happy, surprised and sleepy expressions. "
                "plain white background, korean webtoon style, clean anime line art, flat cel shading, "
                "consistent design across all views, no text, no labels")
ANIMALS = {"corgi"}


def prompt_for(name, desc):
    return (SHEET_ANIMAL if name in ANIMALS else SHEET).format(desc=desc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pick", nargs=2, metavar=("NAME", "SEED"), help="후보 하나를 확정 시트로 복사")
    a = ap.parse_args()

    if a.pick:
        name, seed = a.pick
        src = CAND / f"{name}_s{seed}.png"
        shutil.copy(src, OUT / f"{name}.png")
        print(f"확정: {src.name} → characters/{name}.png")
        return

    story = yaml.safe_load((ROOT / "samples" / "story.yaml").read_text())
    CAND.mkdir(parents=True, exist_ok=True)
    log_path = OUT / "sheets_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else {}

    for name, desc in story["characters"].items():
        if a.only and name not in a.only:
            continue
        prompt = prompt_for(name, desc)
        for seed in a.seeds:
            path = CAND / f"{name}_s{seed}.png"
            if path.exists() and not a.force:
                continue
            t = time.perf_counter()
            subprocess.run([str(MFLUX), "--model", MODEL, "--prompt", prompt, "--seed", str(seed),
                            "--steps", str(a.steps), "--width", str(SIZE[0]), "--height", str(SIZE[1]),
                            "--output", str(path)], check=True)
            alt = path.with_name(path.stem + "_1.png")  # mflux 는 이름 끝에 _1 을 붙여 저장한다
            if alt.exists():
                alt.replace(path)
            sec = time.perf_counter() - t
            log[path.name] = {"character": name, "prompt": prompt, "seed": seed, "steps": a.steps,
                              "size": list(SIZE), "model": MODEL, "sec": round(sec, 1)}
            log_path.write_text(json.dumps(log, ensure_ascii=False, indent=1))
            print(f"{path.name}: {sec:.1f}초", flush=True)


if __name__ == "__main__":
    main()
