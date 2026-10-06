import numpy as np

from scripts.benchmark_instructional_storyboard_clip import (
    choose_threshold,
    metrics,
)


def test_metrics_counts_confusion_matrix() -> None:
    result = metrics(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
    )
    assert result["tp"] == result["fp"] == result["fn"] == result["tn"] == 1


def test_threshold_is_chosen_only_from_precision_constraint() -> None:
    labels = np.array([True, True, False, False])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    threshold = choose_threshold(labels, scores, 0.95)
    result = metrics(labels, scores >= threshold)
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
