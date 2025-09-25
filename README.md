# class‑lm (my tiny LM playground)

This is my attempt at making a language model. Parameters were kept very small compared to todays standards becasuse of the lack of compute power I have. The model was trained on my macbook. Acheived a loss of 1.3 after 6000 steps, but the model and dataset were too small to acheive any coherant results. Though there was some promise, the model was picking some words that were related to the prompt input. 

Pipeline: clean some raw text, train a WordPiece tokenizer, turn text into token bins, and train a tiny GPT with ALiBi attention, then sample.

## Requirements
- Python 3.9+
- PyTorch (`torch`)
- Hugging Face `tokenizers`
- NumPy (`numpy`)
- Optional: `matplotlib` (plots)

Example install:

```bash
pip install torch tokenizers numpy matplotlib
```

On Apple Silicon, training auto‑selects MPS; on NVIDIA, CUDA is used if available.

## What’s inside
- `class-lm/scripts/` – small utilities for cleaning, tokenizing, and plotting
- `class-lm/model/tiny_gpt.py` – Tiny GPT with ALiBi attention
- `class-lm/train/training_stage.py` – training loop + sampling + logging
- `class-lm/train/dset.py` – `PackedBinDataset` for `.bin` token files
- `class-lm/tokenizer/wordpiece/` – tokenizer artifacts (`tokenizer.json`, `vocab.txt`, `tokenizer_config.json`)
- `class-lm/data/` – data and generated bins

## How I run it

1) Clean text (optional but recommended)

```bash
python class-lm/scripts/clean_text_corpus.py \
  --input-dir class-lm/data/domain/text \
  --output-dir class-lm/data/domain/cleaned \
  --auto-remove-repeated-lines \
  --remove-pattern "^Page [0-9]+ of [0-9]+$"
```

Notes:
- `--in-place` overwrites originals instead of writing to `cleaned/`.
- Handy flags: `--collapse-internal-spaces`, `--normalize-ligatures`, `--drop-numeric-lines`, `--join-short-lines`.
- You can also add other patterns to remove.

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

Outputs: `class-lm/tokenizer/wordpiece/tokenizer.json`, `vocab.txt`, and `tokenizer_config.json`.

3) Turn text into a binary token file

```bash
python class-lm/scripts/make_token_bin.py \
  --tokenizer class-lm/tokenizer/wordpiece/tokenizer.json \
  --input-dir class-lm/data/domain/text \
  --extra-dir class-lm/data/generic \
  --output class-lm/data/bins/domain_generic.bin \
  --append-eos
```

- `--glob` controls file discovery (default `*.txt`).
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

Some notes:
- Device picks MPS (Apple) first, then CUDA, then CPU.
- LR scheduling: `--lr-scheduler {cosine,step,poly,none}` (+ warmup flags).
- Resume with `--resume checkpoints/ckpt_<step>.pt`.
- Optional validation: `--val-bin`, `--val-every`, `--val-mb`, `--val-max-batches`.
- Metrics CSV defaults to `checkpoints/metrics.csv` (override with `--metrics-file`).

5) Plot metrics

```bash
python class-lm/scripts/plot_training_metrics.py \
  checkpoints/metrics.csv \
  --output checkpoints/metrics.png
```

## What’s already here (examples)
- Tokenizer: `class-lm/tokenizer/wordpiece/{tokenizer.json,vocab.txt,tokenizer_config.json}`
- Token bins: `class-lm/data/bins/domain_generic.bin`, `class-lm/data/bins/text-corpus.bin`
- Metrics: `loss plot/metrics.png`

## Tips
- Run commands from the repo root so relative paths resolve.
- `PackedBinDataset` must match the `.bin` dtype you used when writing (default `uint16`).
- You can also run as a module: `python -m class-lm.train.training_stage --bin ...`
