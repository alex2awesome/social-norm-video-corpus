import pytest

from scripts.evaluate_commentary_occurred_event_transfer_v3 import evaluate


SHA = "a" * 64


def selected(index, decision):
    return {
        "transfer_index": index,
        "item_id": f"i{index}",
        "uid": f"u{index}",
        "v1_decision": decision,
        "v1_normalized_behavior": "old behavior",
        "v1_normalized_norm": "old norm",
        "source_platform": "youtube",
        "query_source": "seed",
    }


def gold(index, strict):
    return {
        "transfer_index": index,
        "blind_id": f"commentary-transfer-v3-{index:04d}",
        "route": "strict_visual_search" if strict else "none",
        "normalized_behavior": "gold behavior" if strict else "",
        "normalized_norm": "gold norm" if strict else "",
        "description": "manual judgment",
    }


def test_reports_precision_recall_and_every_disagreement():
    selection = [
        selected(0, "accept"),
        selected(1, "accept_after_relabel"),
        selected(2, "reject"),
        selected(3, "reject"),
    ]
    truth = [gold(0, True), gold(1, False), gold(2, True), gold(3, False)]
    report = evaluate(selection, truth, SHA, SHA)
    assert (report["tp"], report["fp"], report["fn"], report["tn"]) == (1, 1, 1, 1)
    assert report["precision"] == 0.5
    assert report["recall"] == 0.5
    assert len(report["disagreements"]) == 2
    assert report["model_v3_evaluated"] is False


def test_rejects_unfrozen_gold_or_coverage_mismatch():
    with pytest.raises(ValueError, match="hash"):
        evaluate([selected(0, "accept")], [gold(0, True)], "b" * 64, SHA)
    with pytest.raises(ValueError, match="coverage"):
        evaluate([selected(0, "accept")], [gold(1, True)], SHA, SHA)
