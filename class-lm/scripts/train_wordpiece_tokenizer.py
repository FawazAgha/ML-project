#!/usr/bin/env python3
"""Train a WordPiece tokenizer on either plain-text files or SQuAD JSON dumps.

By default the script collects every ``.txt`` file stored under
``data/text`` (recursively) and uses the Hugging Face ``tokenizers`` library
to build a WordPiece vocab plus ``tokenizer.json`` bundle. Alternatively you
can point it at SQuAD-style ``.json`` files via ``--squad-train`` (and
optionally ``--squad-dev``). The resulting files are written to
``tokenizer/wordpiece`` by default.
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
    parser.add_argument(
        "--squad-train",
        type=Path,
        help="Path to a SQuAD-style training JSON file (optional).",
    )
    parser.add_argument(
        "--squad-dev",
        type=Path,
        help="Optional SQuAD-style dev JSON file to include in the corpus.",
    )
    args = parser.parse_args()
    if args.squad_dev and not args.squad_train:
        parser.error("--squad-dev requires --squad-train to be set")
    return args


def _extract_texts_from_squad(path: Path, split: str) -> tuple[list[str], dict[str, int]]:
    if not path.exists():
        raise SystemExit(f"SQuAD {split} file '{path}' does not exist")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Failed to parse SQuAD {split} file '{path}': {exc}") from exc

    samples: list[str] = []
    contexts = questions = answers = plausible = 0
    for article in payload.get("data", []):
        for paragraph in article.get("paragraphs", []):
            context = (paragraph.get("context") or "").strip()
            if context:
                samples.append(context)
                contexts += 1
            for qa in paragraph.get("qas", []):
                question = (qa.get("question") or "").strip()
                if question:
                    samples.append(question)
                    questions += 1
                for answer in qa.get("answers", []):
                    text = (answer.get("text") or "").strip()
                    if text:
                        samples.append(text)
                        answers += 1
                for answer in qa.get("plausible_answers", []):
                    text = (answer.get("text") or "").strip()
                    if text:
                        samples.append(text)
                        plausible += 1

    stats = {
        "split": split,
        "path": str(path),
        "contexts": contexts,
        "questions": questions,
        "answers": answers,
        "plausible_answers": plausible,
        "records": len(samples),
    }
    if stats["records"] == 0:
        print(f"Warning: SQuAD {split} file '{path}' yielded zero text entries")
    return samples, stats


def load_squad_texts(train_path: Path, dev_path: Path | None = None) -> tuple[list[str], dict[str, Any]]:
    train_samples, train_stats = _extract_texts_from_squad(train_path, "train")
    all_samples = list(train_samples)
    stats: dict[str, Any] = {
        "splits": {
            "train": train_stats,
        },
    }

    if dev_path is not None:
        dev_samples, dev_stats = _extract_texts_from_squad(dev_path, "dev")
        all_samples.extend(dev_samples)
        stats["splits"]["dev"] = dev_stats

    total_records = sum(split_stats["records"] for split_stats in stats["splits"].values())
    stats["total_records"] = total_records
    stats["num_splits"] = len(stats["splits"])
    stats["paths"] = [split_stats["path"] for split_stats in stats["splits"].values()]
    return all_samples, stats


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
    using_squad = args.squad_train is not None
    if using_squad:
        samples, squad_stats = load_squad_texts(args.squad_train, args.squad_dev)
        print(
            "Training WordPiece on"
            f" {squad_stats['total_records']} text chunks from SQuAD JSON files..."
        )
    else:
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

    if using_squad:
        tokenizer.train_from_iterator(samples, trainer, length=len(samples))
    else:
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
    }
    if using_squad:
        config["corpus"] = {
            "type": "squad",
            "stats": squad_stats,
        }
    else:
        config["corpus"] = {
            "type": "text",
            "num_files": len(files),
            "files": [str(path) for path in files],
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
