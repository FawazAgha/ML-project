#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch Wikipedia pages by category (with recursive subcategories),
focused on CS/DSA (e.g., Category:Algorithms, Category:Data structures).

- Uses wikipedia-api (plain text extraction).
- Deduplicates by title and by paragraph hash.
- Light normalization (strip very short pages, boilerplate-ish tails).
- Respects rate limits with --sleep.
- Stops when target byte size or max pages is reached.

Usage example:
    python fetch_wikipedia_cs.py \
        --categories "Category:Algorithms" "Category:Data structures" \
        --output wiki_cs_corpus.txt \
        --target-bytes 1000000000 \
        --max-depth 3 \
        --sleep 0.2
"""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Set, Dict, Iterable

from tqdm import tqdm
import wikipediaapi


def normalize_whitespace(s: str) -> str:
    s = s.replace('\xa0', ' ')
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


def strip_tail_sections(text: str) -> str:
    """
    Remove common tail sections that are low-value for LM pretraining.
    Works on plain text returned by wikipedia-api.
    """
    # Identify section headers like "References", "External links", "See also"
    # We trim from the first occurrence of these common tails.
    tails = [
        r'\n==\s*References\s*==',
        r'\n==\s*External links\s*==',
        r'\n==\s*Notes\s*==',
        r'\n==\s*Further reading\s*==',
        r'\n==\s*Sources\s*==',
        r'\n==\s*See also\s*==',
    ]
    pattern = re.compile('|'.join(tails), flags=re.IGNORECASE)
    m = pattern.search(text)
    if m:
        return text[:m.start()].rstrip()
    return text


def clean_text(text: str) -> str:
    text = strip_tail_sections(text)
    # Remove footnote-like bracketed numbers e.g., [1], [12]
    text = re.sub(r'\[\d+\]', '', text)
    text = normalize_whitespace(text)
    return text


def hash_paragraph(p: str) -> str:
    return hashlib.sha1(p.encode('utf-8')).hexdigest()


def iter_category_members(wiki, category_page, depth, max_depth) -> Iterable[wikipediaapi.WikipediaPage]:
    """
    DFS over category members, yielding pages.
    """
    if depth > max_depth:
        return
    members = category_page.categorymembers
    for title, member in members.items():
        if member.ns == wikipediaapi.Namespace.CATEGORY:
            yield from iter_category_members(wiki, member, depth + 1, max_depth)
        elif member.ns == wikipediaapi.Namespace.MAIN:
            yield member


def fetch_page_text(page) -> str:
    # page.text is plain text (sections concatenated) with headings like "== Foo =="
    return page.text or ""


def save_checkpoint(path: Path, seen_titles: Set[str], seen_para_hashes: Set[str], stats: Dict):
    ckpt = {
        "seen_titles": list(seen_titles),
        "seen_para_hashes": list(seen_para_hashes),
        "stats": stats,
    }
    path.write_text(json.dumps(ckpt), encoding='utf-8')


def load_checkpoint(path: Path):
    if not path.exists():
        return set(), set(), {"pages_written": 0, "bytes_written": 0}
    data = json.loads(path.read_text(encoding='utf-8'))
    return set(data.get("seen_titles", [])), set(data.get("seen_para_hashes", [])), data.get("stats", {})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", required=True,
                    help='Wikipedia categories, e.g. "Category:Algorithms"')
    ap.add_argument("--output", type=str, required=True, help="Output corpus .txt")
    ap.add_argument("--lang", type=str, default="en", help="Wikipedia language (default: en)")
    ap.add_argument("--user-agent", type=str,
                    default="CS-DSA-Corpus-Bot/1.0 (contact: youremail@example.com)",
                    help="Custom user agent per Wikimedia policy")
    ap.add_argument("--max-depth", type=int, default=3, help="Category recursion depth")
    ap.add_argument("--max-pages", type=int, default=0, help="Stop after N pages (0 = unlimited)")
    ap.add_argument("--target-bytes", type=int, default=0, help="Stop after ~N bytes written (0 = unlimited)")
    ap.add_argument("--min-chars", type=int, default=800, help="Skip pages shorter than this")
    ap.add_argument("--sleep", type=float, default=0.1, help="Sleep seconds between requests")
    ap.add_argument("--checkpoint", type=str, default="", help="Optional checkpoint file to resume")
    ap.add_argument("--checkpoint-every", type=int, default=200, help="Save checkpoint every N pages")
    args = ap.parse_args()

    out_path = Path(args.output)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_path.with_suffix(".ckpt.json")

    wiki = wikipediaapi.Wikipedia(
        language=args.lang,
        user_agent=args.user_agent,
        extract_format=wikipediaapi.ExtractFormat.WIKI,
)

    # Some versions allow: Wikipedia('en', headers={'User-Agent': args.user_agent})

    # Seed category pages
    category_pages = []
    for c in args.categories:
        if not c.lower().startswith("category:"):
            c = "Category:" + c
        cat = wiki.page(c)
        if not cat.exists():
            print(f"[warn] Category not found: {c}", file=sys.stderr)
            continue
        category_pages.append(cat)
    if not category_pages:
        print("[error] No valid categories; exiting.", file=sys.stderr)
        sys.exit(1)

    # Load/prepare state
    seen_titles, seen_para_hashes, stats = load_checkpoint(ckpt_path)
    pages_written = stats.get("pages_written", 0)
    bytes_written = stats.get("bytes_written", 0)

    fout = out_path.open("a", encoding="utf-8")

    # Collect candidate pages (generator per category)
    generators = []
    for cat in category_pages:
        generators.append(iter_category_members(wiki, cat, depth=0, max_depth=args.max_depth))

    # Round-robin over generators to mix categories
    def round_robin(gens):
        gens = list(gens)
        while gens:
            new = []
            for g in gens:
                try:
                    yield next(g)
                    new.append(g)
                except StopIteration:
                    pass
            gens = new

    pbar = tqdm(round_robin(generators), unit="page", desc="Fetching")
    written_this_run = 0
    try:
        for page in pbar:
            if args.max_pages and pages_written >= args.max_pages:
                break
            if args.target_bytes and bytes_written >= args.target_bytes:
                break

            title = page.title
            if title in seen_titles:
                time.sleep(args.sleep)
                continue

            # Fetch
            txt = fetch_page_text(page)
            time.sleep(args.sleep)

            if not txt:
                continue

            txt = clean_text(txt)
            if len(txt) < args.min_chars:
                seen_titles.add(title)
                continue

            # Dedup by paragraph
            paras = [p.strip() for p in txt.split("\n\n") if p.strip()]
            unique_paras = []
            for p in paras:
                # tiny paragraphs are often headings or noise
                if len(p) < 80:
                    continue
                h = hash_paragraph(p.lower())
                if h not in seen_para_hashes:
                    seen_para_hashes.add(h)
                    unique_paras.append(p)

            if not unique_paras:
                seen_titles.add(title)
                continue

            # Write with a simple header delimiter for downstream parsing
            doc = f"### TITLE: {title}\n\n" + "\n\n".join(unique_paras) + "\n\n"
            fout.write(doc)
            fout.flush()

            nbytes = len(doc.encode("utf-8"))
            bytes_written += nbytes
            pages_written += 1
            written_this_run += 1
            seen_titles.add(title)

            pbar.set_postfix(pages=pages_written, mb=round(bytes_written/1e6, 1))

            # Periodic checkpoint
            if written_this_run % args.checkpoint_every == 0:
                save_checkpoint(ckpt_path, seen_titles, seen_para_hashes,
                                {"pages_written": pages_written, "bytes_written": bytes_written})

            # Stop early if targets reached
            if args.max_pages and pages_written >= args.max_pages:
                break
            if args.target_bytes and bytes_written >= args.target_bytes:
                break

    finally:
        fout.close()
        # Final checkpoint
        save_checkpoint(ckpt_path, seen_titles, seen_para_hashes,
                        {"pages_written": pages_written, "bytes_written": bytes_written})

    print(f"[done] Wrote {pages_written} pages, ~{bytes_written/1e6:.1f} MB to {out_path}")
    print(f"[info] Checkpoint saved: {ckpt_path}")


if __name__ == "__main__":
    main()
