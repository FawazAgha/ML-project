#!/usr/bin/env python3
"""Plot training metrics saved by `training_stage.py`."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training metrics CSV")
    parser.add_argument("metrics", type=Path, help="CSV produced by training_stage.py")
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the plot (defaults to metrics directory / metrics.png)",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> tuple[list[int], list[float], list[float]]:
    steps: list[int] = []
    losses: list[float] = []
    ppl: list[float] = []
    with path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            steps.append(int(row["step"]))
            losses.append(float(row["loss"]))
            ppl.append(float(row["perplexity"]))
    if not steps:
        raise SystemExit("No rows found in metrics file")
    return steps, losses, ppl


def plot(metrics_path: Path, output_path: Path) -> None:
    steps, losses, ppl = load_metrics(metrics_path)

    fig, (ax_loss, ax_ppl) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_loss.plot(steps, losses, label="Loss", color="tab:blue")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, linestyle="--", alpha=0.4)

    ax_ppl.plot(steps, ppl, label="Perplexity", color="tab:orange")
    ax_ppl.set_ylabel("Perplexity")
    ax_ppl.set_xlabel("Step")
    ax_ppl.grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    print(f"Saved plot to {output_path}")


def main() -> None:
    args = parse_args()
    metrics_path = args.metrics
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file not found: {metrics_path}")
    output = args.output
    if output is None:
        output = metrics_path.with_suffix(".png")
    plot(metrics_path, output)


if __name__ == "__main__":
    main()
