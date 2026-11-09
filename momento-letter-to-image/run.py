"""CLI로 한 장 생성.

    python run.py --letter "생일 축하해! ..." --event 생일 --style pstl --out out.png
    python run.py --letter_file letter.txt --event birthday --style josen --out out.png
"""

import argparse
import json
from pathlib import Path

import config
from pipeline import MomentoPipeline


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--letter")
    src.add_argument("--letter_file")
    p.add_argument("--event", required=True, help=f"{list(config.EVENTS)} 또는 영어 문구")
    p.add_argument("--style", default="base", choices=list(config.available_styles()))
    p.add_argument("--request_id", help="같은 id면 같은 시드")
    p.add_argument("--out", default="output.png")
    p.add_argument("--show-prompt", action="store_true", help="Print the letter-derived FLUX prompt")
    args = p.parse_args()

    letter = args.letter if args.letter is not None else Path(args.letter_file).read_text(encoding="utf-8")

    result = MomentoPipeline().generate(letter, args.event, args.style, args.request_id)
    result.image.save(args.out)
    print(json.dumps({
        "out": args.out,
        "request_id": result.request_id,
        "seed": result.seed,
        **({"flux_prompt": result.flux_prompt} if args.show_prompt else {}),
        "framing_applied": result.framing_applied,
        "timings": result.timings,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
