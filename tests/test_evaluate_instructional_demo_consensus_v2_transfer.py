from scripts.evaluate_instructional_demo_consensus_v2_transfer import evaluate_transfer


def output(item_id: str, result: dict) -> dict:
    return {"item_id": item_id, "error": None, "result": result}


def v1(value: bool) -> dict:
    return {
        "visual_context": "connected_episode" if value else "presenter_or_interview",
        "participant_grounding": "affected_party_present" if value else "none",
        "physical_social_action": "yes" if value else "no",
        "quote_function": "none_or_unavailable",
    }


def v2(value: bool) -> dict:
    return {"demo_candidate": value}


def prereg() -> dict:
    return {"fresh_validation_gate": {
        "minimum_clips": 2,
        "required_manual_coverage": 2,
        "minimum_precision": 0.8,
        "minimum_recall": 0.4,
        "minimum_selected": 1,
        "source_disjoint_from_all_prior_manual_uids": True,
    }}


def test_transfer_requires_and_records_complete_manual_output_review() -> None:
    manifest = [
        {"audit_index": 0, "item_id": "a", "uid": "ua"},
        {"audit_index": 1, "item_id": "b", "uid": "ub"},
    ]
    manual = [
        {"audit_index": "0", "visual_demo": "Y", "visual_form": "scene"},
        {"audit_index": "1", "visual_demo": "N", "visual_form": "lecture"},
    ]
    v1_rows = [output("a", v1(True)), output("b", v1(True))]
    v2_rows = [output("a", v2(True)), output("b", v2(False))]
    audits = [
        {"audit_index": "0", "v1_output_reviewed": "Y", "v2_output_reviewed": "Y"},
        {"audit_index": "1", "v1_output_reviewed": "Y", "v2_output_reviewed": "Y"},
    ]
    rows, summary = evaluate_transfer(
        manifest, manual, v1_rows, v2_rows, audits, prereg()
    )
    assert [row["v1_v2_consensus"] for row in rows] == [True, False]
    assert summary["metrics"]["v1_v2_consensus"]["precision"] == 1.0
    assert summary["metrics"]["v1_v2_consensus"]["recall"] == 1.0
    assert summary["checks"]["manual_model_output_audit"] is True
    assert summary["preregistered_fresh_gate_passed"] is True


def test_incomplete_output_review_fails_gate() -> None:
    manifest = [
        {"audit_index": 0, "item_id": "a"},
        {"audit_index": 1, "item_id": "b"},
    ]
    manual = [
        {"audit_index": "0", "visual_demo": "Y"},
        {"audit_index": "1", "visual_demo": "N"},
    ]
    rows, summary = evaluate_transfer(
        manifest,
        manual,
        [output("a", v1(True)), output("b", v1(False))],
        [output("a", v2(True)), output("b", v2(False))],
        [{"audit_index": "0", "v1_output_reviewed": "Y", "v2_output_reviewed": "Y"}],
        prereg(),
    )
    assert len(rows) == 2
    assert summary["checks"]["manual_model_output_audit"] is False
    assert summary["preregistered_fresh_gate_passed"] is False
