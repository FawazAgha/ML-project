# class-lm

Minimal end-to-end language modeling pipeline: clean raw text, train a WordPiece tokenizer, convert to token bins, train a tiny GPT with ALiBi attention, and track metrics/samples.

## Requirements
- Python 3.9+
- PyTorch (`torch`)
- Hugging Face `tokenizers`
- NumPy (`numpy`)
- Optional: `matplotlib` (plots), `wordfreq` (generic word list)

Example install:

```bash
pip install torch tokenizers numpy matplotlib wordfreq
```

On Apple Silicon, training auto-selects MPS; on NVIDIA, CUDA is used if available.

## Repository Layout
- `class-lm/scripts/`: utilities for cleaning/conversion/tokenizer training
- `class-lm/model/tiny_gpt.py`: TinyGPT model with ALiBi attention
- `class-lm/train/training_stage.py`: training loop + sampling + logging
- `class-lm/train/dset.py`: `PackedBinDataset` for `.bin` token files
- `class-lm/tokenizer/wordpiece/`: tokenizer artefacts (`tokenizer.json`, `vocab.txt`, `tokenizer_config.json`)
- `class-lm/data/`: corpora and generated bins
- `checkpoints/`: saved checkpoints and metrics

## Typical Workflow

1) Clean converted text (optional but recommended)

```bash
python class-lm/scripts/clean_text_corpus.py \
  --input-dir class-lm/data/domain/text \
  --output-dir class-lm/data/domain/cleaned \
  --auto-remove-repeated-lines \
  --remove-pattern "^Page [0-9]+ of [0-9]+$"
```

Notes:
- `--in-place` overwrites originals instead of writing to `cleaned/`.
- Other helpful flags: `--collapse-internal-spaces`, `--normalize-ligatures`, `--drop-numeric-lines`, `--join-short-lines`.

2) Train a WordPiece tokenizer

```bash
python class-lm/scripts/train_wordpiece_tokenizer.py \
  --data-dir class-lm/data/domain/text \
  --output-dir class-lm/tokenizer/wordpiece \
  --vocab-size 16000 \
  --min-frequency 2 \
  --limit-alphabet 1000 \
  --lowercase
```

Outputs: `class-lm/tokenizer/wordpiece/tokenizer.json`, `vocab.txt`, and `tokenizer_config.json` (settings snapshot).

3) Convert text to a binary token shard

```bash
python class-lm/scripts/make_token_bin.py \
  --tokenizer class-lm/tokenizer/wordpiece/tokenizer.json \
  --input-dir class-lm/data/domain/text \
  --extra-dir class-lm/data/generic \
  --output class-lm/data/bins/domain_generic.bin \
  --append-eos
```

- `--glob` adjusts which files are discovered (default `*.txt`).
- Use `--dtype uint32` if your vocab > 65k tokens.

4) Train TinyGPT

```bash
python class-lm/train/training_stage.py \
  --bin class-lm/data/bins/domain_generic.bin \
  --vocab 16000 \
  --layers 10 --dmodel 512 --heads 8 --ff 2048 --dropout 0.2 \
  --seq 512 --mb 16 --accum 8 \
  --lr 2e-5 --steps 6000 \
  --sample-every 100 --sample-count 5 --sample-max-new 80 \
  --sample-prompt "Explain dynamic arrays" \
  --sample-prompt "What is a heap?" \
  --log-secs 5 \
  --out checkpoints/text-3p6M
```

- Device selection: prefers MPS (Apple), otherwise CUDA if available, else CPU.
- LR scheduling: `--lr-scheduler {cosine,step,poly,none}` (+ warmup options).
- Resume: `--resume checkpoints/ckpt_<step>.pt` restores model/optimizer/scheduler.
- Validation (optional): `--val-bin`, `--val-every`, `--val-mb`, `--val-max-batches`.
- Metrics CSV: `checkpoints/metrics.csv` by default (override with `--metrics-file`).

5) Plot metrics

```bash
python class-lm/scripts/plot_training_metrics.py \
  checkpoints/metrics.csv \
  --output checkpoints/metrics.png
```

## What’s Already Here (examples)
- Tokenizer: `class-lm/tokenizer/wordpiece/{tokenizer.json,vocab.txt,tokenizer_config.json}`
- Token bins: `class-lm/data/bins/domain_generic.bin`, `class-lm/data/bins/text-corpus.bin`
- Metrics: `checkpoints/metrics.csv`, `checkpoints/metrics.png`
- Checkpoint: `checkpoints/text-3p6M/ckpt_6000.pt`

## Tips
- Run commands from the repository root so relative paths resolve.
- `PackedBinDataset` expects the `.bin` dtype you used when writing (default `uint16`).
- You can also run the trainer as a module: `python -m class-lm.train.training_stage --bin ...`
