"""Run with python -m image_error_eval; never calls a model API."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .evaluation import compare, evaluate, load_records, markdown_report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Offline evaluation of saved multi-label image predictions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    comparison = subparsers.add_parser("compare", help="Compare archived runs and constant-label baselines")
    comparison.add_argument("--annotations", type=Path, default=Path("data/annotations.json"))
    comparison.add_argument("--predictions-dir", type=Path, default=Path("results/predictions"))
    comparison.add_argument("--output-dir", type=Path, default=Path("results"))
    single = subparsers.add_parser("evaluate", help="Evaluate one complete saved prediction file")
    single.add_argument("--annotations", type=Path, default=Path("data/annotations.json"))
    single.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            result = compare(args.annotations, args.predictions_dir)
            args.output_dir.mkdir(parents=True, exist_ok=True)
            (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            markdown = markdown_report(result)
            (args.output_dir / "comparison.md").write_text(markdown, encoding="utf-8")
            print(markdown)
        else:
            result = evaluate(load_records(args.annotations, annotations=True), load_records(args.predictions))
            print(json.dumps(result, indent=2))
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
