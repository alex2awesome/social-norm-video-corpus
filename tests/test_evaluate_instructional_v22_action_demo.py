from scripts.evaluate_instructional_v22_action_demo import evaluate


def test_evaluate_v22_model_and_consensus_metrics() -> None:
    manifest = [
        {"audit_index": index, "candidate_id": f"c{index}", "item_id": f"i{index}"}
        for index in range(3)
    ]
    ledger = [
        {
            "audit_index": str(index),
            "candidate_id": f"c{index}",
            "visual_demo": "yes" if index < 2 else "no",
            "visual_form": "scene",
            "blind_visual_note": "note",
        }
        for index in range(3)
    ]
    model_rows = {
        "a": [
            {"item_id": f"i{index}", "error": None, "result": {"demo_pass": "yes" if index != 1 else "no"}}
            for index in range(3)
        ],
        "b": [
            {"item_id": f"i{index}", "error": None, "result": {"demo_pass": "yes" if index < 2 else "no"}}
            for index in range(3)
        ],
    }
    _, summary = evaluate(manifest, ledger, model_rows, expected_count=3)
    assert summary["rules"]["a"]["precision"] == 0.5
    assert summary["rules"]["b"]["precision"] == 1.0
    assert summary["rules"]["all_model_consensus"]["selected"] == 1
    assert summary["rules"]["any_model_positive"]["selected"] == 3
