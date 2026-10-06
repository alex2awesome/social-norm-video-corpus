import pytest

from scripts.evaluate_fresh_scene_scalable_baselines import evaluate


def test_requires_a_successful_score_for_every_item():
    sealed = [
        {
            "audit_index": 0,
            "item_id": "item",
            "uid": "uid",
            "source_modality": "commentary",
        }
    ]
    blind = [{"audit_index": 0, "any_pillar_visual_candidate": "yes"}]
    post = [{"audit_index": 0, "strict_pillar_pass": True}]
    with pytest.raises(ValueError, match="missing successful baseline"):
        evaluate(sealed, blind, post, [])


def test_rejects_manual_coverage_mismatch():
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate(
            [
                {
                    "audit_index": 0,
                    "item_id": "item",
                    "uid": "uid",
                    "source_modality": "commentary",
                }
            ],
            [],
            [],
            [],
        )
