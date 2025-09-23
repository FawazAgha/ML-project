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
  Files whose names contain `Program` repeat at one-sixth the base rate to avoid over-weighting the assorted program snippets.
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

## Converting text to binary shards

Once the tokenizer is ready, turn your text corpus into a binary shard for training:

```bash
python class-lm/scripts/make_token_bin.py \
  --tokenizer class-lm/tokenizer/wordpiece/tokenizer.json \
  --input-dir class-lm/data/domain/text \
  --extra-dir class-lm/data/generic \
  --output class-lm/data/bins/domain_generic.bin
```

Flags of note:
- `--glob` changes which files are picked up (default `*.txt`, recursive).
- `--extra-dir` can be supplied multiple times to fold in additional corpora without repeating `--input-dir`.
- `--append-eos/--no-append-eos` controls whether `[SEP]` gets inserted between files.
- Switch `--dtype` to `uint32` if your tokenizer vocab exceeds 65k entries.

## Training TinyGPT

Launch the training loop (logs, optional sampling, and metrics recording):

```bash
python3 class-lm/train/training_stage.py \
  --bin class-lm/data/bins/domain_generic.bin \
  --vocab 16000 \
  --tokenizer-path class-lm/tokenizer/wordpiece/tokenizer.json \
  --sample-every 50 \
  --sample-count 5 \
  --sample-prompt "Explain dynamic arrays" \
  --sample-prompt "What is a heap?" \
  --log-secs 5 \
  --out checkpoints
```

Key extras:
- Sampling is optional; omit the `--sample-*` flags to disable it.
- Metrics are written to `checkpoints/metrics.csv` (override with `--metrics-file`).
- The CSV captures step, loss, perplexity, tokens/sec, and LR for later analysis.
- Learning rate scheduling: default is cosine. Other options:
  - `--lr-scheduler step` with `--lr-scheduler-step-size` / `--lr-scheduler-gamma`.
  - `--lr-scheduler poly` for polynomial decay (tune aggressiveness via `--lr-scheduler-power`).
  - `--lr-scheduler none` to keep a flat LR (aside from manual `--lr-decay-step`).
- Resume mid-run with `--resume checkpoints/ckpt_<step>.pt` and the trainer will restore the
  model/optimizer/scheduler states and continue from the saved step.

## Plotting training metrics

Render the metrics CSV as a quick loss/perplexity plot:

```bash
pip install matplotlib
python class-lm/scripts/plot_training_metrics.py \
  checkpoints/metrics.csv \
  --output checkpoints/metrics.png
```
