"""Plot and assess training-loss convergence for the FIVES stage runs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_LOGS = {
    "Stages 2": Path("output/train_fives_adaptformer_plus_graph_stages_2/info.log"),
    "Stages 3": Path("output/train_fives_adaptformer_plus_graph_stages_3/info.log"),
    "Stages 2 + 3": Path(
        "output/train_fives_adaptformer_plus_graph_stages_2_3/info.log"
    ),
}

JSON_PATTERN = re.compile(r"\{.*\}")
EPOCH_PATTERN = re.compile(r"Epoch: \[(\d+)\]")
AVERAGED_LOSS_PATTERN = re.compile(r"Averaged stats:\s*.*?loss:\s*[^()]+\(([^()]+)\)")


def parse_log(log_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract one epoch-average training loss for each completed epoch."""
    json_records: dict[int, float] = {}
    averaged_records: dict[int, float] = {}
    current_epoch: int | None = None
    reading_averaged_stats = False

    with log_path.open(encoding="utf-8") as log_file:
        for line in log_file:
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))

            json_match = JSON_PATTERN.search(line)
            if json_match and '"train_loss"' in line:
                record = json.loads(json_match.group(0))
                json_records[int(record["epoch"])] = float(record["train_loss"])
                continue

            if "Averaged stats:" in line:
                reading_averaged_stats = True
                continue

            if reading_averaged_stats and current_epoch is not None:
                loss_match = AVERAGED_LOSS_PATTERN.search(line)
                if loss_match:
                    averaged_records[current_epoch] = float(loss_match.group(1))
                reading_averaged_stats = False

    records = json_records or averaged_records
    if not records:
        raise ValueError(f"No epoch-average training losses found in {log_path}")

    epochs = np.array(sorted(records), dtype=int)
    losses = np.array([records[epoch] for epoch in epochs], dtype=float)
    return epochs, losses


def convergence_summary(epochs: np.ndarray, losses: np.ndarray) -> str:
    """Return a simple, reproducible convergence assessment."""
    if len(losses) < 3:
        return "insufficient data"

    window = max(3, len(losses) // 4)
    tail = losses[-window:]
    slope = float(np.polyfit(epochs, losses, 1)[0])
    tail_slope = float(np.polyfit(epochs[-window:], tail, 1)[0])
    relative_drop = (losses[0] - losses[-1]) / max(abs(losses[0]), 1e-12)
    tail_variation = (tail.max() - tail.min()) / max(abs(tail.mean()), 1e-12)

    if relative_drop >= 0.10 and tail_slope >= -0.01 * losses[0] and tail_variation <= 0.10:
        status = "converged / plateauing"
    elif relative_drop > 0 and slope < 0:
        status = "still decreasing"
    else:
        status = "not converged"

    return (
        f"{status}; first={losses[0]:.4f}, last={losses[-1]:.4f}, "
        f"drop={relative_drop * 100:.1f}%, tail_slope={tail_slope:.5f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training loss and assess convergence for stages 2, 3, and 2+3."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/training_loss_stages_comparison.png"),
        help="Path for the generated plot.",
    )
    args = parser.parse_args()

    figure, axis = plt.subplots(figsize=(10, 6))
    for label, log_path in DEFAULT_LOGS.items():
        if not log_path.exists():
            print(f"{label}: log not found: {log_path}")
            continue

        epochs, losses = parse_log(log_path)
        axis.plot(epochs, losses, marker="o", markersize=3, linewidth=2, label=label)
        print(f"{label}: {convergence_summary(epochs, losses)}")

    axis.set_title("FIVES Training Loss")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Average training loss")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()