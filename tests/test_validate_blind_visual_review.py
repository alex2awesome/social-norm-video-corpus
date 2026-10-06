import pytest

from scripts.validate_blind_visual_review import validate


def review(item_id, scene="no", behavior="no", dense="no"):
    return {
        "item_id": item_id,
        "situated_social_scene": scene,
        "concrete_behavior_visible": behavior,
        "affected_person_or_shared_context_visible": "no",
        "presentation_or_context_only": "yes",
        "technical_or_formal_only": "no",
        "depiction_type": "talking_head",
        "authenticity": "na",
        "dense_review_required": dense,
        "literal_description": "A presenter addresses the camera.",
    }


def test_validate_requires_exact_coverage_and_dense_escalation():
    manifest = [{"item_id": "a"}, {"item_id": "b"}]
    result = validate(
        manifest,
        [review("a"), review("b", scene="uncertain", dense="yes")],
    )
    assert result["coverage"] == 1
    assert result["situated_social_scene"] == {"no": 1, "uncertain": 1}

    with pytest.raises(ValueError, match="requires dense review"):
        validate(manifest, [review("a"), review("b", scene="yes")])
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate(manifest, [review("a")])
