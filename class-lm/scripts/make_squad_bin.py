#!/usr/bin/env python3
"""Convert SQuAD-style JSON into a token .bin for TinyGPT training.

Reads a SQuAD v1.1/v2.0 JSON file and linearises entries into plain text
snippets like:

    question: <Q>\nanswer: <A>\n
Optionally includes the context paragraph. The resulting text is tokenized
with your existing tokenizer.json and written as a flat binary of token IDs
(`uint16` by default) compatible with PackedBinDataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from tokenizers import Tokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert SQuAD JSON to token .bin")
    p.add_argument("--tokenizer", type=Path, required=True, help="tokenizer.json path")
    p.add_argument("--input", type=Path, required=True, help="SQuAD JSON file")
    p.add_argument("--output", type=Path, required=True, help="Destination .bin file")
    p.add_argument(
        "--dtype",
        default="uint16",
        choices=["uint16", "uint32"],
        help="Numeric dtype for tokens (default: uint16)",
    )
    p.add_argument(
        "--include-context",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include paragraph context before Q/A (default: no)",
    )
    p.add_argument(
        "--include-unanswerable",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep unanswerable questions (v2.0) with a placeholder answer",
    )
    p.add_argument(
        "--append-eos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Append [SEP] between examples (default: yes)",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit number of Q/A pairs to convert (0 = all)",
    )
    return p.parse_args()


def iter_squad_examples(path: Path) -> Iterable[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    for article in data.get("data", []):
        for para in article.get("paragraphs", []):
            context = para.get("context", "")
            for qa in para.get("qas", []):
                q = qa.get("question", "").strip()
                is_impossible = qa.get("is_impossible", False)
                answers = qa.get("answers", []) or qa.get("plausible_answers", [])
                a = answers[0]["text"].strip() if answers else ""
                yield {
                    "context": context,
                    "question": q,
                    "answer": a,
                    "is_impossible": bool(is_impossible),
                }


def linearize(ex: dict, include_context: bool, include_unanswerable: bool) -> str | None:
    if ex["is_impossible"] and not include_unanswerable:
        return None
    parts: list[str] = []
    if include_context:
        parts.append(f"context: {ex['context']}")
    parts.append(f"question: {ex['question']}")
    ans = ex["answer"].strip()
    if not ans:
        ans = "unanswerable"
    parts.append(f"answer: {ans}")
    return "\n".join(parts) + "\n"


def main() -> None:
    args = parse_args()
    if not args.tokenizer.exists():
        raise SystemExit(f"Tokenizer not found: {args.tokenizer}")
    if not args.input.exists():
        raise SystemExit(f"Input JSON not found: {args.input}")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    eos_id = tokenizer.token_to_id("[SEP]") or tokenizer.token_to_id("</s>")
    dtype = np.uint16 if args.dtype == "uint16" else np.uint32

    tokens: list[int] = []
    kept = 0
    for i, ex in enumerate(iter_squad_examples(args.input), start=1):
        if args.max_samples and kept >= args.max_samples:
            break
        text = linearize(ex, args.include_context, args.include_unanswerable)
        if text is None:
            continue
        enc = tokenizer.encode(text)
        tokens.extend(enc.ids)
        if args.append_eos and eos_id is not None:
            tokens.append(eos_id)
        kept += 1

    if kept == 0:
        raise SystemExit("No examples were converted (check flags or input file)")

    arr = np.asarray(tokens, dtype=dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(args.output)
    print(
        f"Wrote {len(tokens)} tokens from {kept} examples to {args.output} using dtype {dtype}."
    )


if __name__ == "__main__":
    main()

