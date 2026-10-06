from scripts.evaluate_instructional_v20_semantics import evaluate


def test_evaluate_semantics_counts_exact_relabel_and_recut() -> None:
    visual = [
        {
            "audit_index": index,
            "candidate_id": f"c{index}",
            "sealed_rank_band": (
                "high" if index < 30 else "boundary_below" if index < 45 else "low"
            ),
            "manual_visual_demo": index in {0, 1, 30},
        }
        for index in range(60)
    ]
    semantic = [
        {
            "audit_index": "0",
            "candidate_id": "c0",
            "exact_original": "Y",
            "relabel_usable": "Y",
            "needs_recut": "N",
            "corrected_event": "event zero",
            "failure_mode": "none",
            "semantic_note": "exact",
        },
        {
            "audit_index": "1",
            "candidate_id": "c1",
            "exact_original": "N",
            "relabel_usable": "Y",
            "needs_recut": "Y",
            "corrected_event": "event one",
            "failure_mode": "label_too_broad",
            "semantic_note": "repair",
        },
        {
            "audit_index": "30",
            "candidate_id": "c30",
            "exact_original": "N",
            "relabel_usable": "N",
            "needs_recut": "N",
            "corrected_event": "",
            "failure_mode": "not_social",
            "semantic_note": "bad",
        },
    ]
    rows, summary = evaluate(visual, semantic)
    assert summary["overall"]["visual_demos"] == 3
    assert summary["overall"]["exact_original_labels"] == 1
    assert summary["overall"]["relabel_usable"] == 2
    assert summary["overall"]["needs_recut"] == 1
    assert summary["bands"]["high"]["relabel_usable"] == 2
    assert rows[59]["semantic_failure_mode"] == "visual_non_demo"


def test_evaluate_requires_every_visual_positive() -> None:
    visual = [
        {
            "audit_index": index,
            "candidate_id": f"c{index}",
            "sealed_rank_band": "high",
            "manual_visual_demo": index == 0,
        }
        for index in range(60)
    ]
    try:
        evaluate(visual, [])
    except ValueError as exc:
        assert "cover every visual positive" in str(exc)
    else:
        raise AssertionError("expected incomplete semantic ledger to fail")
