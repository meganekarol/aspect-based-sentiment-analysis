"""Command-line interface: ``absa analyze "The pizza was great" --model lexicon``."""

from __future__ import annotations

import argparse
import json
import sys

from absa import get_analyzer

MODELS = ("lexicon", "transformer", "llm")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="absa", description="Aspect-Based Sentiment Analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="Extract aspect-sentiment pairs from text")
    p.add_argument("text", nargs="?", help="Text to analyze (reads stdin when omitted)")
    p.add_argument("-m", "--model", choices=MODELS, default="lexicon", help="Back-end (default: lexicon)")
    p.add_argument("--llm-model", default="llama3", help="Ollama model name for --model llm")
    p.add_argument("--json", action="store_true", help="Print results as JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    text = args.text if args.text is not None else sys.stdin.read()
    if not text.strip():
        print("error: no text given", file=sys.stderr)
        return 2

    kwargs = {"model_name": args.llm_model} if args.model == "llm" else {}
    results = get_analyzer(args.model, **kwargs).analyze(text)

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2))
    elif not results:
        print("No aspects detected.")
    else:
        for r in results:
            print(f"{r.aspect:<25} {r.sentiment:<9} {r.confidence:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
