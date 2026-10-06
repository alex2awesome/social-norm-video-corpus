#!/usr/bin/env python3
"""Shared evaluation metrics for scoping experiments (pure python).

AUROC via rank statistic, accuracy at a threshold, Wilson intervals, and
per-slice breakdowns.  Calibration reuses the label-model reliability bins.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from weaksup.lf_matrix_v1 import wilson_lower


def auroc(scores: list[float], labels: list[int]) -> float | None:
    """Probability a random positive outscores a random negative (ties=0.5)."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must align")
    positives = [s for s, y in zip(scores, labels) if y == 1]
    negatives = [s for s, y in zip(scores, labels) if y == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for p in positives:
        for n in negatives:
            wins += 1.0 if p > n else 0.5 if p == n else 0.0
    return wins / (len(positives) * len(negatives))


def accuracy_at(scores: list[float], labels: list[int], threshold: float) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for score, label in zip(scores, labels):
        predicted = score >= threshold
        if predicted and label == 1: tp += 1
        elif predicted: fp += 1
        elif label == 1: fn += 1
        else: tn += 1
    n = tp + fp + fn + tn
    correct = tp + tn
    return {
        "threshold": threshold, "n": n,
        "accuracy": correct / n if n else None,
        "accuracy_wilson_lower": wilson_lower(correct, n),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def sliced_auroc(
    rows: list[dict[str, Any]], score_key: str, label_key: str, slice_key: str
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(slice_key))].append(row)
    return {
        name: {
            "n": len(group),
            "auroc": auroc([r[score_key] for r in group],
                           [r[label_key] for r in group]),
        }
        for name, group in sorted(groups.items())
        if len(group) >= 20
    }
