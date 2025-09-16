# ML-project
## Cleaning converted text files

Use `scripts/clean_text_corpus.py` to tidy `.txt` files produced from PDFs.

Example workflow:

```bash
python scripts/clean_text_corpus.py \
  --input-dir class-lm/data/domain/text \
  --output-dir class-lm/data/domain/cleaned \
  --auto-remove-repeated-lines \
  --remove-pattern "^Page [0-9]+ of [0-9]+$"
```

Key flags:
- `--in-place` overwrites the originals instead of writing to `cleaned/`.
- `--collapse-internal-spaces` aggressively collapses whitespace within lines (leave off for code snippets).
- `--dry-run` reports changes without touching the files, useful to validate the heuristics first.
