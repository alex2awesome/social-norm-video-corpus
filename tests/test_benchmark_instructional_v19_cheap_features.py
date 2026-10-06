import numpy as np

from scripts.benchmark_instructional_v19_cheap_features import (
    best_precision_threshold,
    choose_recall_floor,
    keyed_scores,
)


def test_recall_floor_is_highest_threshold_preserving_recall() -> None:
    labels = np.asarray([True, True, False, False])
    values = np.asarray([0.1, 0.3, 0.2, 0.4])
    assert choose_recall_floor(labels, values, 0.5) == 0.3
    assert choose_recall_floor(labels, values, 1.0) == 0.1


def test_keyed_scores_merges_successful_codec_retry(tmp_path) -> None:
    first = tmp_path / "first.jsonl"
    retry = tmp_path / "retry.jsonl"
    first.write_text(
        '{"item_id":"a","error":"codec"}\n'
        '{"item_id":"b","low_level":{"motion_mean":1}}\n'
    )
    retry.write_text(
        '{"item_id":"a","low_level":{"motion_mean":2}}\n'
    )
    merged = keyed_scores([first, retry])
    assert set(merged) == {"a", "b"}
    assert merged["a"]["low_level"]["motion_mean"] == 2


def test_best_precision_threshold_requires_support() -> None:
    labels = np.asarray([True, False, True, False])
    scores = np.asarray([0.9, 0.8, 0.7, 0.1])
    threshold, result = best_precision_threshold(labels, scores, 2)
    assert threshold == 0.7
    assert result["accepted"] == 3
    assert result["precision"] == 2 / 3
