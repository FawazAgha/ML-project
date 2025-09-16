#!/usr/bin/env python3
"""Train a WordPiece tokenizer over a corpus of plain-text files.

The script relies on the Hugging Face `tokenizers` library and mirrors the
WordPiece configuration used by BERT-style models. Provide an input directory
containing `.txt` documents and it will produce a `tokenizer.json` plus
`vocab.txt` artefacts that the rest of the pipeline can load.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

try:
    from tokenizers import Tokenizer
    from tokenizers import normalizers, pre_tokenizers, processors
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
except ImportError as exc:  # pragma: no cover - guidance for the user.
    raise SystemExit(
        "tokenizers package not found. Install it with 'pip install tokenizers'."
    ) from exc


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a WordPiece tokenizer")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/domain/text"),
        help="Directory containing cleaned .txt files (recursively processed).",
    )
    parser.add_argument(
        "--extra-dir",
        action="append",
        type=Path,
        default=[],
        metavar="PATH",
        help=(
            "Additional directory of .txt files to include once (can be supplied "
            "multiple times)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tokenizer/wordpiece"),
        help="Directory to store tokenizer outputs (tokenizer.json, vocab.txt, ...).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16000,
        help="Target vocabulary size (including special tokens).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Discard tokens seen fewer times than this during training.",
    )
    parser.add_argument(
        "--limit-alphabet",
        type=int,
        default=1000,
        help="Maximum distinct initial characters to keep before using unknown token.",
    )
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply lowercase normalisation before tokenisation (default: enabled).",
    )
    parser.add_argument(
        "--repeat-input",
        type=int,
        default=1,
        help=(
            "Repeat the list of discovered files this many times during training. "
            "Useful for up-weighting a small domain corpus."
        ),
    )
    return parser.parse_args()


def iter_text_files(base_dir: Path) -> Iterable[Path]:
    if not base_dir.exists():
        raise SystemExit(f"Input directory '{base_dir}' does not exist")
    for path in sorted(base_dir.rglob("*.txt")):
        if path.is_file():
            yield path


def prepare_tokenizer(lowercase: bool) -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    normalizer_steps: List[normalizers.Normalizer] = [normalizers.NFD(), normalizers.StripAccents()]
    if lowercase:
        normalizer_steps.append(normalizers.Lowercase())
    tokenizer.normalizer = normalizers.Sequence(normalizer_steps)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    return tokenizer


def main() -> None:
    args = parse_args()
    primary_paths = [path for path in iter_text_files(args.input_dir)]
    if not primary_paths:
        raise SystemExit(f"No .txt files found under '{args.input_dir}'.")
    if args.repeat_input < 1:
        raise SystemExit("--repeat-input must be at least 1")
    files: list[str] = []
    for path in primary_paths:
        repeat_count = args.repeat_input
        if args.repeat_input == 1:
            repeat_count = 1
        elif "program" in path.name.lower():
            repeat_count = max(1, args.repeat_input // 6)
        files.extend([str(path)] * repeat_count)

    extra_dirs = []
    for directory in args.extra_dir:
        extra_files = [str(path) for path in iter_text_files(directory)]
        if not extra_files:
            raise SystemExit(f"No .txt files found under extra dir '{directory}'.")
        files.extend(extra_files)
        extra_dirs.append(str(directory))

    tokenizer = prepare_tokenizer(lowercase=args.lowercase)
    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix="##",
        limit_alphabet=args.limit_alphabet,
    )

    print(
        "Training WordPiece on "
        f"{len(files)} file references (repeat={args.repeat_input})..."
    )
    tokenizer.train(files, trainer)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_json = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))

    vocab_files = tokenizer.model.save(str(output_dir))
    vocab_paths = [Path(path) for path in vocab_files]

    config = {
        "lowercase": args.lowercase,
        "input_dir": str(args.input_dir),
        "repeat_input": args.repeat_input,
        "extra_dirs": extra_dirs,
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "limit_alphabet": args.limit_alphabet,
        "special_tokens": SPECIAL_TOKENS,
        "tokenizer_json": str(tokenizer_json),
        "vocab_files": [str(path) for path in vocab_paths],
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"Saved tokenizer JSON -> {tokenizer_json}")
    for path in vocab_paths:
        print(f"Saved vocab -> {path}")
    print("Done.")


if __name__ == "__main__":
    main()
