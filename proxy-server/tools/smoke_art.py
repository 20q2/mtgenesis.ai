"""
Generate card art for one prompt and save it to data/smoke.png.

Usage (from proxy-server/):
    <PY> tools/smoke_art.py "an ancient red dragon atop a volcano" [--colors R] [--type Creature] [--runs 3]

Prints the elapsed seconds per run. Run 1 includes loading the model (and, the very
first time, downloading it); later runs in the same process are warm.
"""
import argparse
import base64
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import config  # noqa: E402
import image_generation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prompt")
    parser.add_argument("--colors", default="R", help="color letters, e.g. R or BG ('' for colorless)")
    parser.add_argument("--type", default="Creature")
    parser.add_argument("--runs", type=int, default=1, help="generate this many times; runs after the first are warm")
    args = parser.parse_args()

    card_data = {"colors": list(args.colors), "type": args.type}
    out = Path(config.DATA_DIR) / "smoke.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    for run in range(1, args.runs + 1):
        start = time.perf_counter()
        art_b64 = image_generation.generate_art(args.prompt, card_data)
        elapsed = time.perf_counter() - start
        out.write_bytes(base64.b64decode(art_b64))
        label = "cold (includes model load)" if run == 1 else "warm"
        print(f"run {run}: {elapsed:.2f}s [{label}]")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
