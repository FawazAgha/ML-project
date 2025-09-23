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
python class-lm/scripts/clean_text_corpus.py \
    --input-dir class-lm/data/text \
    --output-dir class-lm/data/text_clean \
    --auto-remove-repeated-lines \
    --remove-pattern "^Page [0-9]+ of [0-9]+$"
"""

from __future__ import annotations

import argparse
import keyword
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

# Project root for sensible defaults
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class CleanStats:
    """Book-keeping about the transformations applied."""

    hyphen_merges: int = 0
    blank_lines_removed: int = 0
    repeated_lines_removed: int = 0
    pattern_lines_removed: int = 0
    whitespace_collapses: int = 0
    numeric_lines_removed: int = 0

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
        default=PROJECT_ROOT / "data" / "text",
        help="Directory containing source .txt files (default: class-lm/data/text).",
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
        "--normalize-ligatures",
        action="store_true",
        help=(
            "Normalize common PDF ligatures and smart quotes/dashes to ASCII (e.g., ﬁ→fi, “→\" , —→-)."
        ),
    )
    parser.add_argument(
        "--drop-numeric-lines",
        action="store_true",
        help=(
            "Remove lines that contain only numbers (optionally separated by whitespace). "
            "Useful for cleaning stray page numbers or numbered lists detached from context."
        ),
    )
    parser.add_argument(
        "--numeric-line-max-tokens",
        type=int,
        default=1,
        help=(
            "When dropping numeric-only lines, drop lines with up to this many numeric tokens "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--drop-complexity-lines",
        action="store_true",
        help=(
            "Also treat tokens like 'N', 'n', 'N2', 'n^2' as numeric-like; lines composed only of "
            "these tokens (up to --numeric-line-max-tokens) will be removed."
        ),
    )
    parser.add_argument(
        "--drop-python-keyword-lines",
        action="store_true",
        help=(
            "Remove lines composed solely of Python keywords (e.g., 'for', 'while', 'class')."
        ),
    )
    parser.add_argument(
        "--keyword-line-max-tokens",
        type=int,
        default=3,
        help="Drop keyword-only lines with up to this many tokens (default: 3).",
    )
    parser.add_argument(
        "--strip-dot-leaders",
        action="store_true",
        help=(
            "Strip trailing dot leaders from headings (e.g., 'Types......' -> 'Types') and drop lines "
            "that are only dots."
        ),
    )
    parser.add_argument(
        "--join-short-lines",
        action="store_true",
        help=(
            "Join consecutive short title-like lines into one (e.g., 'The If and Switch' + 'Statements' -> 'The If and Switch Statements')."
        ),
    )
    parser.add_argument(
        "--join-short-lines-max-len",
        type=int,
        default=40,
        help="Maximum length of a line to consider for joining (default: 40).",
    )
    parser.add_argument(
        "--relax-hyphenation-merge",
        action="store_true",
        help=(
            "Merge hyphenated line breaks even when the next line starts with an uppercase letter (e.g., 'Pseudo-' + 'Code' -> 'Pseudo-Code')."
        ),
    )
    parser.add_argument(
        "--drop-language-keyword-lines",
        action="append",
        choices=["python", "java"],
        default=[],
        help=(
            "Additionally drop lines that are solely language keywords for the given "
            "language(s). Can be passed multiple times (e.g., --drop-language-keyword-lines java)."
        ),
    )
    parser.add_argument(
        "--keyword-list-file",
        type=Path,
        help=(
            "Optional path to a file containing extra tokens (whitespace-separated). Lines "
            "composed only of these tokens (up to --keyword-line-max-tokens) will be removed."
        ),
    )
    parser.add_argument(
        "--fix-spaced-letter-runs",
        action="store_true",
        help=(
            "Collapse in-line runs of single lowercase letters separated by spaces (e.g., 'l o g' -> 'log')."
        ),
    )
    parser.add_argument(
        "--merge-single-letter-lines",
        action="store_true",
        help=(
            "Merge consecutive lines that are a single lowercase letter each into a single word; "
            "if the preceding line ends with '=' or '(', attach the merged word to that line."
        ),
    )
    parser.add_argument(
        "--min-letter-run",
        type=int,
        default=3,
        help="Minimum length of letter runs to merge (default: 3).",
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


def normalize_ligatures_and_quotes(text: str) -> str:
    """Best-effort replacement of common PDF ligatures and smart punctuation.

    This aims to improve tokenization quality on OCR/PDF text.
    """
    replacements = {
        # Latin ligatures
        "": "ff",   # not common, placeholder kept for completeness
        "": "ffi",  # placeholder
        "": "ffl",  # placeholder
        "i": "fi",  # ﬁ
        "l": "fl",  # ﬂ
        # Dashes
        "": "-",    # en dash (–)
        "": "-",    # em dash (—)
        # Quotes
        "": '"',   # left double (“)
        "": '"',   # right double (”)
        "": "'",   # left single (‘)
        "": "'",   # right single (’)
    }
    # Fallback using explicit code points to avoid font surprises
    replacements.update({
        "ﬁ": "fi",
        "ﬂ": "fl",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    })
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


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


def undo_hyphenation(lines: Sequence[str], *, allow_uppercase_next: bool = False) -> tuple[List[str], int]:
    merged: List[str] = []
    merges = 0
    i = 0
    while i < len(lines):
        current = lines[i]
        stripped = current.rstrip()
        if stripped.endswith("-") and i + 1 < len(lines):
            next_line = lines[i + 1]
            next_stripped = next_line.lstrip()
            cond = next_stripped and (next_stripped[0].islower() or (allow_uppercase_next and next_stripped[0].isalpha()))
            if cond:
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


def remove_numeric_only_lines(
    lines: Sequence[str],
    *,
    max_tokens: int = 1,
    include_complexity: bool = False,
) -> tuple[List[str], int]:
    """Drop lines that are numeric-only or consist of simple complexity tokens.

    - Numeric tokens accept thousands separators (e.g., "1,000", "100,000,000").
    - If ``include_complexity`` is True, tokens like "N", "n", "N2", "n^2" are also
      considered numeric-like and thus droppable when they appear alone on a line.
    - ``max_tokens`` caps how many such tokens are allowed on a line for it to be
      removed (to avoid deleting large numeric tables).
    """
    cleaned: List[str] = []
    removed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        # Tokenize on whitespace and validate each token.
        tokens = stripped.split()
        if 0 < len(tokens) <= max_tokens:
            all_simple = True
            for t in tokens:
                # Plain integer with optional commas (e.g., 1,000,000)
                if re.fullmatch(r"\d[\d,]*", t):
                    continue
                # Optional: complexity tokens like N, N2, n^3
                if include_complexity and re.fullmatch(r"[Nn](?:\^?\d+)?", t):
                    continue
                all_simple = False
                break
            if all_simple:
                removed += 1
                continue
        cleaned.append(line)
    return cleaned, removed


def strip_dot_leaders_lines(lines: Sequence[str]) -> List[str]:
    cleaned: List[str] = []
    for line in lines:
        s = line.rstrip()
        # Drop lines that are only dot leaders (with or without spaces between dots)
        if re.fullmatch(r"\s*(?:\.|\s)*(?:\.(?:\s*\.){2,})\s*", s) or \
           re.fullmatch(r"\s*(?:\.(?:\s*\.){2,})\s*", s):
            continue
        # Strip trailing dot leaders: patterns like '...',' . . . . . '
        s = re.sub(r"\s*(?:\.(?:\s*\.){2,})\s*$", "", s)
        cleaned.append(s)
    return cleaned


def join_short_title_lines(lines: Sequence[str], *, max_len: int = 40) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        if i + 1 < len(lines):
            nxt = lines[i + 1].strip()
            if 0 < len(cur) <= max_len and 0 < len(nxt) <= max_len:
                # Heuristic: current line does not end with sentence punctuation; next starts with capital
                if not re.search(r"[\.;:!?)]$", cur) and (nxt[:1].isupper() or nxt[:1].isdigit()):
                    out.append((cur + " " + nxt).rstrip())
                    i += 2
                    continue
        out.append(lines[i])
        i += 1
    return out


def remove_keyword_only_lines(
    lines: Sequence[str], *, max_tokens: int = 3, keywords: set[str] | None = None
) -> tuple[List[str], int]:
    if keywords is None:
        keywords = set(keyword.kwlist)
    cleaned: List[str] = []
    removed = 0
    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned.append(line)
            continue
        tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", stripped)
        if 0 < len(tokens) <= max_tokens and all(t in keywords for t in tokens):
            removed += 1
            continue
        cleaned.append(line)
    return cleaned, removed


def java_keywords() -> set[str]:
    """Return a set of Java language keywords and literals (Java 17+)."""
    return {
        # Keywords
        "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
        "class", "const", "continue", "default", "do", "double", "else", "enum",
        "extends", "final", "finally", "float", "for", "goto", "if", "implements",
        "import", "instanceof", "int", "interface", "long", "native", "new",
        "package", "private", "protected", "public", "return", "short", "static",
        "strictfp", "super", "switch", "synchronized", "this", "throw", "throws",
        "transient", "try", "void", "volatile", "while",
        # Newer (records/sealed/yield/var)
        "record", "sealed", "permits", "non-sealed", "yield", "var",
        # Literals
        "true", "false", "null",
    }


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


def collapse_spaced_letter_runs(line: str) -> str:
    """Collapse in-line runs of lowercase letters separated by spaces into a word.

    Example: "l o g n" -> "logn"; leaves uppercase sequences (e.g., 'U S A') untouched.
    """
    def repl(match: re.Match[str]) -> str:
        letters = match.group(0).split()
        if all(ch.islower() and len(ch) == 1 for ch in letters):
            return "".join(letters)
        return match.group(0)

    # Replace sequences of at least 3 single-letter tokens
    return re.sub(r"\b(?:[a-z])(?:\s+[a-z]){2,}\b", repl, line)


def merge_single_letter_lines(lines: Sequence[str], *, min_run: int = 3) -> List[str]:
    """Merge consecutive single-lowercase-letter lines into one word.

    If the previous non-empty output line ends with '=' or '(', attach the merged
    word to that line separated by a space; otherwise, emit it as a standalone line.
    """
    out: List[str] = []
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if len(s) == 1 and s.islower():
            j = i
            letters: List[str] = []
            while j < len(lines):
                sj = lines[j].strip()
                if len(sj) == 1 and sj.islower():
                    letters.append(sj)
                    j += 1
                else:
                    break
            if len(letters) >= min_run:
                word = "".join(letters)
                # Attach to previous if it ends with '=' or '('
                if out:
                    prev = out[-1].rstrip()
                    if prev.endswith(("=", "(")):
                        out[-1] = prev + " " + word
                    else:
                        out.append(word)
                else:
                    out.append(word)
                i = j
                continue
        # default: keep line as-is
        out.append(lines[i])
        i += 1
    return out


def clean_text(
    text: str,
    *,
    max_blank_lines: int,
    remove_patterns: Sequence[re.Pattern[str]],
    auto_remove: bool,
    min_repeated_count: int,
    max_repeated_length: int,
    collapse_spaces: bool,
    allow_uppercase_next: bool = False,
) -> tuple[str, CleanStats]:
    stats = CleanStats()
    normalised = normalize_newlines(text)
    # Optional ligature/quote normalization occurs in main() based on CLI flags.
    lines = normalised.split("\n")

    lines = remove_trailing_spaces(lines)

    lines, merges = undo_hyphenation(lines, allow_uppercase_next=allow_uppercase_next)
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

    # Optionally remove numeric-only lines at the end so earlier heuristics
    # (e.g. header/footer detection) can see the original structure.
    if auto_remove is not None:  # no-op to satisfy lints about unused arg
        pass
    # This feature is controlled via CLI flags; evaluated in main() below.

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
        if args.normalize_ligatures:
            original = normalize_ligatures_and_quotes(original)
        cleaned, stats = clean_text(
            original,
            max_blank_lines=args.max_blank_lines,
            remove_patterns=patterns,
            auto_remove=args.auto_remove_repeated_lines,
            min_repeated_count=args.min_repeated_line_count,
            max_repeated_length=args.max_repeated_line_length,
            collapse_spaces=args.collapse_internal_spaces,
            allow_uppercase_next=args.relax_hyphenation_merge,
        )

        # Fix in-line spaced letter runs like "l o g" -> "log"
        if args.fix_spaced_letter_runs and cleaned:
            tmp_lines = cleaned.split("\n")
            tmp_lines = [collapse_spaced_letter_runs(ln) for ln in tmp_lines]
            cleaned = "\n".join(tmp_lines)
            cleaned = cleaned.strip("\n")
            if cleaned:
                cleaned += "\n"

        # Strip TOC dot leaders and join split headings
        if cleaned and (args.strip_dot_leaders or args.join_short_lines):
            tmp = cleaned.split("\n")
            if args.strip_dot_leaders:
                tmp = strip_dot_leaders_lines(tmp)
            if args.join_short_lines:
                tmp = join_short_title_lines(tmp, max_len=max(10, args.join_short_lines_max_len))
            cleaned = "\n".join(tmp)
            cleaned = cleaned.strip("\n")
            if cleaned:
                cleaned += "\n"

        # Post-pass: drop numeric-only lines if requested.
        if (args.drop_numeric_lines or args.drop_complexity_lines) and cleaned:
            lines = cleaned.split("\n")
            lines, removed_numeric = remove_numeric_only_lines(
                lines,
                max_tokens=max(1, args.numeric_line_max_tokens),
                include_complexity=args.drop_complexity_lines,
            )
            if removed_numeric:
                stats.numeric_lines_removed = removed_numeric
                cleaned = "\n".join(lines)
                cleaned = cleaned.strip("\n")
                if cleaned:
                    cleaned += "\n"

        # Merge consecutive single-letter lines into words
        if args.merge_single_letter_lines and cleaned:
            lines = cleaned.split("\n")
            lines = merge_single_letter_lines(lines, min_run=max(2, args.min_letter_run))
            cleaned = "\n".join(lines)
            cleaned = cleaned.strip("\n")
            if cleaned:
                cleaned += "\n"

        # Remove keyword-only lines (Python/Java/custom lists)
        if cleaned:
            combined_keywords: set[str] = set()
            if args.drop_python_keyword_lines:
                combined_keywords.update(keyword.kwlist)
            for lang in args.drop_language_keyword_lines:
                if lang == "python":
                    combined_keywords.update(keyword.kwlist)
                elif lang == "java":
                    combined_keywords.update(java_keywords())
            if args.keyword_list_file is not None and args.keyword_list_file.exists():
                try:
                    extra = args.keyword_list_file.read_text(encoding="utf-8").split()
                    combined_keywords.update(extra)
                except Exception:
                    pass
            if combined_keywords:
                lines = cleaned.split("\n")
                lines, removed_kw = remove_keyword_only_lines(
                    lines,
                    max_tokens=max(1, args.keyword_line_max_tokens),
                    keywords=combined_keywords,
                )
                if removed_kw:
                    cleaned = "\n".join(lines)
                    cleaned = cleaned.strip("\n")
                    if cleaned:
                        cleaned += "\n"

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
