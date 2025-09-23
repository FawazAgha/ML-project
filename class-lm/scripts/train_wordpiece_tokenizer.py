#!/usr/bin/env python3
"""Train a WordPiece tokenizer on a folder of plain-text files.

The script collects every ``.txt`` file under ``data/text`` (recursively) and
uses the Hugging Face ``tokenizers`` library to build a WordPiece vocab plus
``tokenizer.json`` bundle. Artefacts are written to ``tokenizer/wordpiece`` by
default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from tokenizers import Tokenizer
    from tokenizers import decoders
    from tokenizers import normalizers, pre_tokenizers, processors
    from tokenizers.models import WordPiece
    from tokenizers.trainers import WordPieceTrainer
except ImportError as exc:  # pragma: no cover - user guidance only.
    raise SystemExit(
        "tokenizers package not found. Install it with 'pip install tokenizers'."
    ) from exc


SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "text"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "tokenizer" / "wordpiece"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a WordPiece tokenizer on data/text corpus"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing plain-text training files (default: data/text).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for tokenizer artefacts (default: tokenizer/wordpiece).",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=16000,
        help="Target vocabulary size including special tokens (default: 16000).",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency required to stay in the vocab (default: 2).",
    )
    parser.add_argument(
        "--limit-alphabet",
        type=int,
        default=1000,
        help="Maximum distinct initial characters kept before using [UNK] (default: 1000).",
    )
    parser.add_argument(
        "--lowercase",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply lowercase normalisation (default: enabled).",
    )
    return parser.parse_args()




def discover_corpus(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise SystemExit(f"Corpus directory '{data_dir}' does not exist")
    files = sorted(path for path in data_dir.rglob("*.txt") if path.is_file())
    if not files:
        raise SystemExit(f"No .txt files found under '{data_dir}'.")
    return files


def build_tokenizer(lowercase: bool) -> Tokenizer:
    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    normalizer_steps: list[normalizers.Normalizer] = [
        normalizers.NFD(),
        normalizers.StripAccents(),
    ]
    if lowercase:
        normalizer_steps.append(normalizers.Lowercase())
    tokenizer.normalizer = normalizers.Sequence(normalizer_steps)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer.decoder = decoders.WordPiece(prefix="##")
    return tokenizer


def train_tokenizer(args: argparse.Namespace) -> None:
    files = discover_corpus(args.data_dir)
    print(f"Training WordPiece on {len(files)} files from '{args.data_dir}'...")
    tokenizer = build_tokenizer(lowercase=args.lowercase)
    trainer = WordPieceTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=SPECIAL_TOKENS,
        continuing_subword_prefix="##",
        limit_alphabet=args.limit_alphabet,
    )

    tokenizer.train([str(path) for path in files], trainer)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_json = output_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_json))
    vocab_files = tokenizer.model.save(str(output_dir))
    vocab_paths = [Path(path) for path in vocab_files]

    config: dict[str, Any] = {
        "data_dir": str(args.data_dir),
        "lowercase": args.lowercase,
        "vocab_size": args.vocab_size,
        "min_frequency": args.min_frequency,
        "limit_alphabet": args.limit_alphabet,
        "tokenizer_json": str(tokenizer_json),
        "vocab_files": [str(path) for path in vocab_paths],
        "corpus": {
            "type": "text",
            "num_files": len(files),
            "files": [str(path) for path in files],
        },
    }
    (output_dir / "tokenizer_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"Saved tokenizer JSON -> {tokenizer_json}")
    for path in vocab_paths:
        print(f"Saved vocab -> {path}")
    print("Done.")


def main() -> None:
    args = parse_args()
    train_tokenizer(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    main()
