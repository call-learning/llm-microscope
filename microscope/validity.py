"""Confidence, calibration, and controlled-variant helpers.

These metrics describe empirical reliability. They do not determine whether a
model output is factually correct without an external label or reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def prediction_uncertainty(logits: torch.Tensor, target_id: int | None = None) -> dict[str, float | int]:
    """Return confidence metrics for one next-token logit vector."""
    values = logits.detach().float().flatten()
    probabilities = values.softmax(dim=-1)
    top_values, top_ids = probabilities.topk(min(2, probabilities.numel()))
    chosen_id = int(top_ids[0]) if target_id is None else int(target_id)
    if chosen_id < 0 or chosen_id >= probabilities.numel():
        raise ValueError("Target token ID is outside the vocabulary.")
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
    return {
        "target_id": chosen_id,
        "probability": float(probabilities[chosen_id]),
        "entropy": float(entropy),
        "top1_id": int(top_ids[0]),
        "top1_probability": float(top_values[0]),
        "top2_probability": float(top_values[1]) if len(top_values) > 1 else 0.0,
        "top1_margin": float(top_values[0] - top_values[1]) if len(top_values) > 1 else 1.0,
        "surprisal": float(-probabilities[chosen_id].clamp_min(1e-12).log()),
    }


def calibration_metrics(probabilities: list[float] | np.ndarray, labels: list[int] | np.ndarray,
                         bins: int = 10) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Calculate reliability metrics and per-bin calibration data.

    ``probabilities`` are confidence values for the predicted class and
    ``labels`` are 1 when that prediction is correct, otherwise 0.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probabilities.ndim != 1 or labels.ndim != 1 or len(probabilities) != len(labels):
        raise ValueError("Probabilities and labels must be equally sized one-dimensional arrays.")
    if not len(probabilities):
        raise ValueError("At least one labelled prediction is required.")
    if np.any((probabilities < 0) | (probabilities > 1)) or np.any((labels < 0) | (labels > 1)):
        raise ValueError("Probabilities and labels must be in the range 0..1.")
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1], right=False), bins - 1)
    rows = []
    ece = 0.0
    for index in range(bins):
        selected = assignments == index
        count = int(selected.sum())
        confidence = float(probabilities[selected].mean()) if count else np.nan
        accuracy = float(labels[selected].mean()) if count else np.nan
        if count:
            ece += count / len(labels) * abs(confidence - accuracy)
        rows.append({
            "bin": index,
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "confidence": confidence,
            "accuracy": accuracy,
        })
    brier = float(np.mean((probabilities - labels) ** 2))
    metrics = {
        "n_examples": len(labels),
        "accuracy": float(labels.mean()),
        "expected_calibration_error": float(ece),
        "brier_score": brier,
        "majority_baseline": float(max(labels.mean(), 1 - labels.mean())),
    }
    return metrics, pd.DataFrame(rows)


def compare_variants(records: list[dict[str, object]]) -> pd.DataFrame:
    """Validate and tabulate explicit prompt-variant result records."""
    if not records:
        raise ValueError("At least one variant record is required.")
    required = {"variant", "probability", "token", "entropy"}
    missing = required - set(records[0])
    if missing:
        raise ValueError(f"Variant records are missing: {', '.join(sorted(missing))}.")
    frame = pd.DataFrame(records)
    frame["probability"] = pd.to_numeric(frame["probability"], errors="raise")
    frame["entropy"] = pd.to_numeric(frame["entropy"], errors="raise")
    frame["agreement_with_first"] = frame["token"] == frame.iloc[0]["token"]
    return frame
