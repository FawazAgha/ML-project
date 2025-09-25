# Training Loss Summary

- Plot: `checkpoints/metrics.png`
- Source: `checkpoints/metrics.csv`

## Highlights
- First logged step: 120
  - loss ≈ 4.7001, perplexity ≈ 109.96, lr ≈ 4.06e-05
- Best (lowest) loss: step 2565
  - loss ≈ 1.1835, perplexity ≈ 3.2658
- Final logged step: 6000
  - loss ≈ 1.3003, perplexity ≈ 3.6704, lr ≈ 5.00e-05
- Logged points: 392 (loss reported roughly every ~15 steps)
- Throughput stays stable around ~16–17k tokens/sec

## Interpretation
- Early phase (≈ steps 120–600): rapid loss drop as the model learns token statistics and short-range structure from the corpus. This is typical for next-token cross-entropy on a reasonably tokenized dataset.
- Mid phase (≈ steps 600–2600): continued but slowing improvement as the model captures longer-range dependencies. The minimum training loss occurs around step 2565.
- Late phase (≈ steps 2600–6000): small fluctuations and a slight drift upward in the average training loss compared to the minimum. This can happen due to:
  - stochastic variation from minibatch sampling, dropout (default 0.2), and accumulation windows
  - the learning-rate schedule not being at a local optimum for the final stretch
  - mild overfitting signals on the training metric if regularization or schedule choices differ from the dataset scale

## Why this curve makes sense
- Tokenization: WordPiece with `[CLS]/[SEP]` boundaries produces clean token targets, enabling a fast initial loss decline.
- Model bias: TinyGPT with ALiBi handles variable context lengths without positional embedding drift, helping early convergence.
- Optimization: AdamW with a moderate LR produces a smooth curve; the recorded LR (~4e-5→1e-4→5e-5 over the run) aligns with warmup/plateau behavior that can yield a low mid-run minimum.

## Suggestions
- Add/enable validation to confirm generalization: pass a validation `.bin` via `--val-bin` and log `val_loss`/`val_ppl`.
- If the goal is a lower final training loss:
  - Try a short warmup (`--warmup-steps 200–500`) and cosine/linear decay tuned to total steps.
  - Slightly lower dropout (e.g., `--dropout 0.1`) if under-regularized data isn’t a concern.
  - Increase steps or tokens (bigger `.bin`) if the curve hasn’t plateaued.
- If the goal is better generalization:
  - Keep dropout at 0.2, consider label smoothing (e.g., `--label-smoothing 0.05`), and monitor validation metrics.

## Files
- Metrics CSV: `checkpoints/metrics.csv`
- Plot: `checkpoints/metrics.png`
- Example checkpoint: `checkpoints/text-3p6M/ckpt_6000.pt`
