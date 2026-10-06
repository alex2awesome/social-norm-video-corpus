from scripts.evaluate_instructional_v20_rank_holdout import evaluate


def test_evaluate_reveals_band_metrics_after_join() -> None:
    sealed = [
        {
            "audit_index": index,
            "candidate_id": f"c{index}",
            "sealed_rank_band": (
                "high" if index < 30 else "boundary_below" if index < 45 else "low"
            ),
        }
        for index in range(60)
    ]
    ledger = [
        {
            "audit_index": str(index),
            "candidate_id": f"c{index}",
            "visual_demo": "Y" if index < 10 else "N",
            "visual_conclusive": "yes",
            "visual_form": "scene" if index < 10 else "talking_head",
            "blind_visual_note": "note",
        }
        for index in range(60)
    ]
    _, summary = evaluate(sealed, ledger)
    assert summary["overall"]["visual_demos"] == 10
    assert summary["bands"]["high"]["visual_demo_ratio"] == 1 / 3
    assert summary["bands"]["low"]["visual_demo_ratio"] == 0


def test_evaluate_supports_frozen_expansion_sample_and_yes_labels() -> None:
    sealed = [
        {
            "audit_index": index,
            "candidate_id": f"c{index}",
            "sealed_rank_band": ("high", "boundary_below", "low")[index % 3],
        }
        for index in range(30)
    ]
    ledger = [
        {
            "audit_index": str(index),
            "candidate_id": f"c{index}",
            "visual_demo": "yes" if index in {0, 3, 6} else "no",
            "visual_conclusive": "yes",
            "visual_form": "scene" if index in {0, 3, 6} else "talking_head",
            "blind_visual_note": "note",
        }
        for index in range(30)
    ]
    _, summary = evaluate(sealed, ledger, expected_count=30)
    assert summary["overall"]["visual_demos"] == 3
    assert summary["bands"]["high"]["visual_demo_ratio"] == 0.3
