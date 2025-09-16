#!/usr/bin/env python3
"""Utility for tidying text that originated from PDF extractions.

The script performs a couple of conservative clean-up passes:
- normalises line endings to ``\n``
- trims trailing whitespace and reduces long blank-line runs
- optionally detects repeated header/footer lines and removes them
- optionally removes user supplied patterns
- merges words that were hyphen-split across line breaks

Example
-------
python scripts/clean_text_corpus.py \
    --input-dir data/domain/text \
    --output-dir data/domain/cleaned \
    --auto-remove-repeated-lines \
    --remove-pattern "^Page [0-9]+ of [0-9]+$"
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass
class CleanStats:
    """Book-keeping about the transformations applied."""

    hyphen_merges: int = 0
    blank_lines_removed: int = 0
    repeated_lines_removed: int = 0
    pattern_lines_removed: int = 0
    whitespace_collapses: int = 0

    def changed(self) -> bool:
        return any(
            getattr(self, field) > 0
            for field in (
                "hyphen_merges",
                "blank_lines_removed",
                "repeated_lines_removed",
                "pattern_lines_removed",
                "whitespace_collapses",
            )
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean converted PDF text files.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/domain/text"),
        help="Directory containing source .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Where to write cleaned files. Defaults to '<input-dir>/cleaned'. "
            "Set --in-place to overwrite the originals."
        ),
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite the input files instead of writing to a new directory.",
    )
    parser.add_argument(
        "--max-blank-lines",
        type=int,
        default=1,
        help="Limit of consecutive blank lines to allow (default: 1).",
    )
    parser.add_argument(
        "--auto-remove-repeated-lines",
        action="store_true",
        help="Drop short lines that repeat many times (heuristic header/footer removal).",
    )
    parser.add_argument(
        "--min-repeated-line-count",
        type=int,
        default=5,
        help="Minimum occurrences before a line is treated as a repeated header/footer.",
    )
    parser.add_argument(
        "--max-repeated-line-length",
        type=int,
        default=80,
        help="Ignore repeated-line detection for longer lines (default: 80).",
    )
    parser.add_argument(
        "--remove-pattern",
        action="append",
        default=[],
        metavar="REGEX",
        help="Regex pattern for lines to drop (can be supplied multiple times).",
    )
    parser.add_argument(
        "--collapse-internal-spaces",
        action="store_true",
        help="Collapse consecutive whitespace within non-empty lines (safer off for code).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report which files would change without writing results.",
    )

    args = parser.parse_args()
    if args.in_place and args.output_dir:
        parser.error("--in-place cannot be combined with --output-dir")
    if not args.input_dir.exists():
        parser.error(f"Input directory '{args.input_dir}' does not exist")
    return args


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def normalize_newlines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.lstrip("\ufeff")


def remove_trailing_spaces(lines: Sequence[str]) -> List[str]:
    return [line.rstrip() for line in lines]


def collapse_blank_lines(lines: Sequence[str], max_blank_lines: int) -> tuple[List[str], int]:
    collapsed: List[str] = []
    blank_run = 0
    removed = 0
    for line in lines:
        if not line.strip():
            blank_run += 1
            if blank_run > max_blank_lines:
                removed += 1
                continue
        else:
            blank_run = 0
        collapsed.append(line)
    return collapsed, removed


def undo_hyphenation(lines: Sequence[str]) -> tuple[List[str], int]:
    merged: List[str] = []
    merges = 0
    i = 0
    while i < len(lines):
        current = lines[i]
        stripped = current.rstrip()
        if stripped.endswith("-") and i + 1 < len(lines):
            next_line = lines[i + 1]
            next_stripped = next_line.lstrip()
            if next_stripped and next_stripped[0].islower():
                merged.append(stripped[:-1] + next_stripped)
                merges += 1
                i += 2
                continue
        merged.append(stripped)
        i += 1
    return merged, merges


def compile_patterns(patterns: Sequence[str]) -> List[re.Pattern[str]]:
    compiled: List[re.Pattern[str]] = []
    for raw in patterns:
        try:
            compiled.append(re.compile(raw))
        except re.error as exc:
            raise SystemExit(f"Invalid regex '{raw}': {exc}") from exc
    return compiled


def remove_pattern_lines(lines: Sequence[str], patterns: Sequence[re.Pattern[str]]) -> tuple[List[str], int]:
    if not patterns:
        return list(lines), 0
    cleaned: List[str] = []
    removed = 0
    for line in lines:
        stripped = line.strip()
        if any(pattern.search(stripped) for pattern in patterns):
            removed += 1
            continue
        cleaned.append(line)
    return cleaned, removed


def normalize_for_repeat_detection(line: str) -> str:
    stripped = line.strip()
    stripped = re.sub(r"\s+", " ", stripped)
    stripped = re.sub(r"\d+", "#", stripped)
    return stripped.lower()


def auto_remove_repeated_lines(
    lines: Sequence[str],
    min_count: int,
    max_length: int,
) -> tuple[List[str], int]:
    counter: Counter[str] = Counter()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if len(stripped) > max_length:
            continue
        normalized = normalize_for_repeat_detection(stripped)
        if not normalized:
            continue
        if normalized.isdigit():
            continue
        counter[normalized] += 1
    targets = {norm for norm, count in counter.items() if count >= min_count}
    if not targets:
        return list(lines), 0

    cleaned: List[str] = []
    removed = 0
    for line in lines:
        normalized = normalize_for_repeat_detection(line)
        if normalized in targets:
            removed += 1
            continue
        cleaned.append(line)
    return cleaned, removed


def collapse_internal_spaces(lines: Sequence[str]) -> tuple[List[str], int]:
    collapsed: List[str] = []
    changes = 0
    for line in lines:
        if not line.strip():
            collapsed.append(line)
            continue
        compact = re.sub(r"\s+", " ", line.strip())
        if compact != line:
            changes += 1
        collapsed.append(compact)
    return collapsed, changes


def clean_text(
    text: str,
    *,
    max_blank_lines: int,
    remove_patterns: Sequence[re.Pattern[str]],
    auto_remove: bool,
    min_repeated_count: int,
    max_repeated_length: int,
    collapse_spaces: bool,
) -> tuple[str, CleanStats]:
    stats = CleanStats()
    normalised = normalize_newlines(text)
    lines = normalised.split("\n")

    lines = remove_trailing_spaces(lines)

    lines, merges = undo_hyphenation(lines)
    stats.hyphen_merges = merges

    lines, removed_by_pattern = remove_pattern_lines(lines, remove_patterns)
    stats.pattern_lines_removed = removed_by_pattern

    if auto_remove:
        lines, removed_auto = auto_remove_repeated_lines(
            lines,
            min_count=min_repeated_count,
            max_length=max_repeated_length,
        )
        stats.repeated_lines_removed = removed_auto

    lines, blank_removed = collapse_blank_lines(lines, max_blank_lines)
    stats.blank_lines_removed = blank_removed

    if collapse_spaces:
        lines, collapsed = collapse_internal_spaces(lines)
        stats.whitespace_collapses = collapsed

    cleaned = "\n".join(lines)
    cleaned = cleaned.strip("\n")
    if cleaned:
        cleaned += "\n"
    return cleaned, stats


def determine_output_path(path: Path, input_dir: Path, output_dir: Path | None, in_place: bool) -> Path:
    if in_place:
        return path
    if output_dir is None:
        output_dir = input_dir / "cleaned"
    relative = path.relative_to(input_dir)
    destination = output_dir / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def iter_text_files(base_dir: Path) -> Iterable[Path]:
    for candidate in sorted(base_dir.rglob("*.txt")):
        if candidate.is_file():
            yield candidate


def main() -> None:
    args = parse_args()
    patterns = compile_patterns(args.remove_pattern)

    for path in iter_text_files(args.input_dir):
        original = load_text(path)
        cleaned, stats = clean_text(
            original,
            max_blank_lines=args.max_blank_lines,
            remove_patterns=patterns,
            auto_remove=args.auto_remove_repeated_lines,
            min_repeated_count=args.min_repeated_line_count,
            max_repeated_length=args.max_repeated_line_length,
            collapse_spaces=args.collapse_internal_spaces,
        )

        if not stats.changed() and cleaned == original:
            continue

        destination = determine_output_path(
            path,
            args.input_dir,
            args.output_dir,
            in_place=args.in_place,
        )

        if args.dry_run:
            print(f"[DRY RUN] {path} -> {destination} (changes: {stats})")
            continue

        destination.write_text(cleaned, encoding="utf-8")
        print(f"Updated {destination} ({stats})")


if __name__ == "__main__":
    main()
