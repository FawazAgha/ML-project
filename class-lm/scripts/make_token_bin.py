#!/usr/bin/env python3
"""Tokenize a text corpus and dump tokens into a binary file."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer

DEFAULT_PATTERN = "*.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tokenize text files into a binary shard")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        required=True,
        help="Path to tokenizer.json produced earlier.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing text files to tokenize (recursively scanned).",
    )
    parser.add_argument(
        "--extra-dir",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help="Additional directory to include once (can be passed multiple times).",
    )
    parser.add_argument(
        "--glob",
        default=DEFAULT_PATTERN,
        help=f"Glob pattern for files inside input-dir (default: {DEFAULT_PATTERN}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination .bin file; will be overwritten if it exists.",
    )
    parser.add_argument(
        "--dtype",
        default="uint16",
        choices=["uint16", "uint32"],
        help="Numeric dtype for tokens (default: uint16; use uint32 for larger vocabs).",
    )
    parser.add_argument(
        "--append-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append an EOS token between files (default: yes).",
    )
    return parser.parse_args()


def iter_files(base: Path, pattern: str) -> Iterable[Path]:
    for path in sorted(base.rglob(pattern)):
        if path.is_file():
            yield path


def main() -> None:
    args = parse_args()
    if not args.tokenizer.exists():
        raise SystemExit(f"Tokenizer file not found: {args.tokenizer}")
    if not args.input_dir.exists():
        raise SystemExit(f"Input directory not found: {args.input_dir}")
    for extra in args.extra_dir:
        if not extra.exists():
            raise SystemExit(f"Extra directory not found: {extra}")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_token_id = tokenizer.token_to_id("[SEP]") or tokenizer.token_to_id("</s>")

    dtype = np.uint16 if args.dtype == "uint16" else np.uint32
    tokens: list[int] = []

    files = list(iter_files(args.input_dir, args.glob))
    for extra in args.extra_dir:
        files.extend(iter_files(extra, args.glob))
    if not files:
        raise SystemExit("No files found to tokenize.")

    for path in files:
        text = path.read_text(encoding="utf-8")
        encoded = tokenizer.encode(text)
        tokens.extend(encoded.ids)
        if args.append_eos and eos_token_id is not None:
            tokens.append(eos_token_id)

    arr = np.asarray(tokens, dtype=dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(args.output)
    print(f"Wrote {len(tokens)} tokens to {args.output} using dtype {dtype}.")


if __name__ == "__main__":
    main()
