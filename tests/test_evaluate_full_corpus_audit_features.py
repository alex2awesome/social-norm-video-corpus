import numpy as np

from scripts.evaluate_full_corpus_audit_features import (
    final_visual_label,
    metrics,
)


def test_final_visual_label_uses_sparse_when_dense_not_required():
    sparse = {
        "dense_review_required": "no",
        "situated_social_scene": "yes",
        "concrete_behavior_visible": "yes",
    }
    assert final_visual_label("item", sparse, {}) == 1


def test_final_visual_label_uses_dense_when_required():
    sparse = {
        "dense_review_required": "yes",
        "situated_social_scene": "no",
        "concrete_behavior_visible": "no",
    }
    dense = {
        "item": {
            "dense_scene_visible": "yes",
            "dense_concrete_action_visible": "yes",
        }
    }
    assert final_visual_label("item", sparse, dense) == 1


def test_metrics_reports_expected_confusion_matrix():
    result = metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0.1, 0.8, 0.7, 0.2]),
    )
    assert result["confusion_matrix"] == [[1, 1], [1, 1]]
    assert result["precision_at_0_5"] == 0.5
    assert result["recall_at_0_5"] == 0.5
