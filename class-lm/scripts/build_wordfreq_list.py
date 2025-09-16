#!/usr/bin/env python3
"""Dump the top-N words from `wordfreq` into the generic corpus folder."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from wordfreq import top_n_list
except ImportError as exc:  # pragma: no cover - guidance for the user.
    raise SystemExit("wordfreq not installed. Run 'pip install wordfreq'.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a high-frequency word list")
    parser.add_argument(
        "--language",
        default="en",
        help="Language code passed to wordfreq (default: en).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=50000,
        help="Number of most frequent words to output (default: 50000).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generic/freqword.txt"),
        help="Output path for the newline-separated word list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    words = top_n_list(args.language, args.top_n)
    if not words:
        raise SystemExit("No words returned; check language code or top-n value.")

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(words)} words from language '{args.language}' to {output_path}"
    )


if __name__ == "__main__":
    main()
