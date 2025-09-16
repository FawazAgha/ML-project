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

## Training a WordPiece tokenizer

1. Install dependencies (ideally in a virtualenv):
 ```bash
  pip install tokenizers
  ```
2. Feed the cleaned corpus into the trainer:
   ```bash
python class-lm/scripts/train_wordpiece_tokenizer.py \
  --input-dir class-lm/data/domain/text \
  --output-dir class-lm/tokenizer/wordpiece \
  --vocab-size 16000 \
  --repeat-input 20 \
  --extra-dir class-lm/data/generic
   ```

Key flags:
- `--lowercase/--no-lowercase` toggles lowercasing during normalisation.
- `--repeat-input` duplicates the discovered files to up-weight a smaller corpus.
- `--extra-dir` can be passed multiple times to include additional corpora once (e.g., generic text).
  `generic.txt` inside the main `--input-dir` is automatically repeated at half the rate of the other files.
- `--min-frequency` and `--limit-alphabet` mirror Hugging Face defaults.
- Point `--input-dir` at any folder of `.txt` files if you organise datasets differently.

Outputs land in the chosen `--output-dir` as `tokenizer.json`, `vocab.txt`, and a `tokenizer_config.json` snapshot of the settings used.

## Building a high-frequency word list

Generate a supplementary generic word list (requires `wordfreq`):

```bash
pip install wordfreq
python class-lm/scripts/build_wordfreq_list.py \
  --language en \
  --top-n 50000 \
  --output class-lm/data/generic/freqword.txt
```

Adjust the `--language` code or `--top-n` size if you need a different distribution.
