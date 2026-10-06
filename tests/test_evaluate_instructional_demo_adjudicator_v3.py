import pytest

from scripts.evaluate_instructional_demo_adjudicator_v3 import evaluate


def test_evaluate_requires_complete_review_and_computes_metrics() -> None:
    manifest = [
        {"audit_index": 0, "item_id": "a"},
        {"audit_index": 1, "item_id": "b"},
    ]
    manual = [
        {"audit_index": "0", "visual_demo": "Y"},
        {"audit_index": "1", "visual_demo": "N"},
    ]
    outputs = [
        {"item_id": "a", "error": None, "result": {"demo_candidate": True}},
        {"item_id": "b", "error": None, "result": {"demo_candidate": False}},
    ]
    reviews = [
        {"audit_index": "0", "output_reviewed": "Y", "output_visually_supported": "yes"},
        {"audit_index": "1", "output_reviewed": "Y", "output_visually_supported": "no", "error_mechanism": "wrong_behavior"},
    ]
    rows, summary = evaluate(manifest, manual, outputs, reviews)
    assert len(rows) == 2
    assert summary["metric"]["precision"] == 1.0
    assert summary["metric"]["recall"] == 1.0
    assert summary["manual_output_review_complete"] is True
    assert summary["model_output_visual_support_rate"] == 0.5
    assert summary["manual_error_mechanisms"] == {"wrong_behavior": 1}


def test_evaluate_fails_closed_on_coverage_mismatch() -> None:
    with pytest.raises(ValueError, match="coverage mismatch"):
        evaluate(
            [{"audit_index": 0, "item_id": "a"}],
            [{"audit_index": "0", "visual_demo": "Y"}],
            [{"item_id": "a", "error": None, "result": {"demo_candidate": True}}],
            [],
        )


def test_evaluate_accepts_new_manifest_and_manual_ledger_shapes() -> None:
    rows, summary = evaluate(
        [{"storyboard_index": 7, "item_id": "a"}],
        [{"audit_index": "7", "manual_visual_demo": "yes"}],
        [{"item_id": "a", "error": None, "result": {"demo_candidate": True}}],
        [{"audit_index": "7", "output_reviewed": "yes", "output_visually_supported": "yes", "note": "supported"}],
        kind="replication",
        status="known_gold_replication_not_promotion_eligible",
    )
    assert rows[0]["audit_index"] == 7
    assert rows[0]["manual_visual_demo"] is True
    assert rows[0]["manual_output_reviewed"] is True
    assert summary["kind"] == "replication"
    assert summary["status"] == "known_gold_replication_not_promotion_eligible"


def test_evaluate_allows_manual_ledger_superset_for_render_failures() -> None:
    rows, summary = evaluate(
        [{"storyboard_index": 7, "item_id": "a"}],
        [
            {"audit_index": "6", "manual_visual_demo": "no"},
            {"audit_index": "7", "manual_visual_demo": "yes"},
        ],
        [{"item_id": "a", "error": None, "result": {"demo_candidate": True}}],
        [{"audit_index": "7", "output_reviewed": "yes", "output_visually_supported": "yes"}],
    )
    assert len(rows) == 1
    assert summary["manual_outputs_reviewed"] == 1


def test_evaluate_rejects_invalid_new_manual_label() -> None:
    with pytest.raises(ValueError, match="invalid manual visual-demo label"):
        evaluate(
            [{"storyboard_index": 7, "item_id": "a"}],
            [{"audit_index": "7", "manual_visual_demo": "maybe"}],
            [{"item_id": "a", "error": None, "result": {"demo_candidate": True}}],
            [{"audit_index": "7", "output_reviewed": "yes", "output_visually_supported": "yes"}],
        )


def test_evaluate_requires_visual_support_for_every_output() -> None:
    with pytest.raises(ValueError, match="lacks visual-support adjudication"):
        evaluate(
            [{"audit_index": 0, "item_id": "a"}],
            [{"audit_index": "0", "visual_demo": "Y"}],
            [{"item_id": "a", "error": None, "result": {"demo_candidate": True}}],
            [{"audit_index": "0", "output_reviewed": "yes"}],
        )
