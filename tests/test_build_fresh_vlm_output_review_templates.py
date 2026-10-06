import pytest

from scripts.build_fresh_vlm_output_review_templates import build


def test_instructional_template_exposes_prediction_after_exact_coverage():
    fields, rows = build(
        "instructional",
        [{"item_id": "i", "uid": "u"}],
        [{"item_id": "i", "uid": "u", "model": "m", "result": {"demo_candidate": True}, "error": None}],
    )
    assert "rationale_supported" in fields
    assert rows[0]["model_demo_candidate"] == "yes"
    assert rows[0]["structured_prediction_supported"] == ""


def test_witnessed_template_requires_prompt_version():
    with pytest.raises(ValueError, match="prompt_version"):
        build(
            "witnessed",
            [{"candidate_id": "c", "uid": "u"}],
            [{"candidate_id": "c", "uid": "u", "model": "m", "result": {}, "error": None}],
        )


def test_commentary_template_records_strict_prediction():
    _, rows = build(
        "commentary",
        [{"candidate_id": "c", "uid": "u"}],
        [{"candidate_id": "c", "model": "m", "strict_pass": False, "error": None}],
    )
    assert rows[0]["model_strict_pass"] == "no"


def test_rejects_partial_or_failed_output_coverage():
    with pytest.raises(ValueError, match="exactly cover"):
        build("instructional", [{"item_id": "i", "uid": "u"}], [])
    with pytest.raises(ValueError, match="contain errors"):
        build(
            "instructional",
            [{"item_id": "i", "uid": "u"}],
            [{"item_id": "i", "uid": "u", "model": "m", "error": "timeout"}],
        )


def test_rejects_model_output_uid_lineage_mismatch():
    with pytest.raises(ValueError, match="uid lineage mismatch"):
        build(
            "instructional",
            [{"item_id": "i", "uid": "u"}],
            [{
                "item_id": "i", "uid": "other", "model": "m",
                "result": {"demo_candidate": True}, "error": None,
            }],
        )
