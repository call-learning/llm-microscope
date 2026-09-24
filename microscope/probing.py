from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from .analysis import analyse_batch
from .runtime import Runtime


@dataclass
class ProbeResult:
    per_layer_accuracy: pd.DataFrame
    direction: dict[int, np.ndarray]
    n_prompts: int
    n_positive: int
    n_negative: int
    evaluation: str
    baseline_accuracy: float
    heldout_supported: bool


def generate_concept_labels(
    pos_template: str,
    neg_template: str,
    fill_values: str,
) -> tuple[list[str], list[int]]:
    """Generate a balanced labeled set from contrastive templates.

    Args:
        pos_template: Template with ``{X}`` placeholder for positive prompts.
        neg_template: Template with ``{X}`` placeholder for negative prompts.
        fill_values: Comma-separated fill values (e.g. "France, Germany, Japan").

    Returns:
        (prompts, labels) where prompts alternates pos/neg per fill value
        and labels are 1 (pos) or 0 (neg).
    """
    values = [v.strip() for v in fill_values.split(",") if v.strip()]
    if not values:
        raise ValueError("At least one fill value is required.")
    prompts: list[str] = []
    labels: list[int] = []
    for val in values:
        prompts.append(pos_template.replace("{X}", val))
        labels.append(1)
        prompts.append(neg_template.replace("{X}", val))
        labels.append(0)
    return prompts, labels


def generate_quick_labels(
    prompts: list[str],
    substring: str,
) -> list[int]:
    """Label prompts by substring presence.

    Args:
        prompts: List of prompt strings.
        substring: The substring to search for.

    Returns:
        List of 1/0 labels (1 if substring is in the prompt).
    """
    return [1 if substring in p else 0 for p in prompts]


def fit_probes(
    runtime: Runtime,
    prompts: list[str],
    labels: list[int],
    C: float = 1.0,
) -> ProbeResult:
    """Fit a binary L2 logistic regression probe on the final-token hidden
    state at each decoder layer.

    Uses in-sample evaluation: the classifier is fit and evaluated on the
    same data (no train/val split).

    Args:
        runtime: Loaded model runtime.
        prompts: Training prompts.
        labels: Binary labels (0/1), same length as prompts.
        C: Inverse of the L2 penalty strength (higher = less regularization).

    Returns:
        ProbeResult with per-layer accuracy and the learned direction vector
        for each layer.
    """
    if len(prompts) != len(labels):
        raise ValueError(
            f"prompts ({len(prompts)}) and labels ({len(labels)}) must be the same length."
        )
    if len(set(labels)) < 2:
        raise ValueError("Need at least one positive and one negative example.")

    result = analyse_batch(runtime, prompts)

    n_layers = len(runtime.layers)
    accuracies: list[float] = []
    train_accuracies: list[float] = []
    heldout_accuracies: list[float] = []
    heldout_stds: list[float] = []
    directions: dict[int, np.ndarray] = {}

    y = np.array(labels)
    class_counts = np.bincount(y, minlength=2)
    can_holdout = len(prompts) >= 6 and int(class_counts.min()) >= 2
    split: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    if can_holdout:
        try:
            split = train_test_split(
                np.arange(len(y)),
                y,
                test_size=0.25,
                random_state=42,
                stratify=y,
            )
        except ValueError:
            can_holdout = False
    repeated_splits = can_holdout and len(prompts) >= 12 and int(class_counts.min()) >= 4
    evaluation = "held-out stratified evaluation" if can_holdout else "training-only evaluation"
    baseline_accuracy = float(max(y.mean(), 1 - y.mean()))

    for layer_idx in range(n_layers):
        # hidden_states[0] is embedding output; hidden_states[L+1] is after layer L
        hidden = result.hidden_states[layer_idx + 1]
        # Final-token hidden state per prompt: shape (batch, hidden)
        X = hidden[:, -1, :].numpy()
        clf = LogisticRegression(C=C, max_iter=1000)
        clf.fit(X, y)
        train_accuracy = float(clf.score(X, y))
        train_accuracies.append(train_accuracy)
        if split is not None:
            train_indices, test_indices, _, _ = split
            split_clf = LogisticRegression(C=C, max_iter=1000)
            split_clf.fit(X[train_indices], y[train_indices])
            heldout_accuracy = float(split_clf.score(X[test_indices], y[test_indices]))
            repeated_scores = [heldout_accuracy]
            if repeated_splits:
                splitter = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=42)
                repeated_scores = []
                for train_indices, test_indices in splitter.split(X, y):
                    repeated_clf = LogisticRegression(C=C, max_iter=1000)
                    repeated_clf.fit(X[train_indices], y[train_indices])
                    repeated_scores.append(float(repeated_clf.score(X[test_indices], y[test_indices])))
            accuracies.append(heldout_accuracy)
            heldout_accuracies.append(heldout_accuracy)
            heldout_stds.append(float(np.std(repeated_scores)) if len(repeated_scores) > 1 else float("nan"))
        else:
            accuracies.append(train_accuracy)
            heldout_accuracies.append(float("nan"))
            heldout_stds.append(float("nan"))
        directions[layer_idx] = clf.coef_[0].copy()

    per_layer_accuracy = pd.DataFrame(
        {
            "layer": list(range(n_layers)),
            "accuracy": accuracies,
            "train_accuracy": train_accuracies,
            "heldout_accuracy": heldout_accuracies,
            "baseline_accuracy": baseline_accuracy,
            "heldout_std": heldout_stds,
            "evaluation": evaluation,
        }
    )

    return ProbeResult(
        per_layer_accuracy=per_layer_accuracy,
        direction=directions,
        n_prompts=len(prompts),
        n_positive=int(sum(labels)),
        n_negative=int(len(labels) - sum(labels)),
        evaluation=evaluation,
        baseline_accuracy=baseline_accuracy,
        heldout_supported=can_holdout,
    )
