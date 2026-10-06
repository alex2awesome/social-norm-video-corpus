import pytest

from scripts.evaluate_instructional_evidence_bands import evaluate, load_gold


def test_candidate_metrics_count_relabels_as_usable():
    bands = [
        {"item_id": "a", "band": "blind_dual_exact_candidate"},
        {"item_id": "b", "band": "blind_dual_relabel_candidate"},
        {"item_id": "c", "band": "blind_dual_exact_candidate"},
        {"item_id": "d", "band": "low_evidence_review"},
    ]
    gold = {
        "a": {"decision": "accept"},
        "b": {"decision": "accept_after_relabel"},
        "c": {"decision": "reject"},
        "d": {"decision": "accept"},
    }

    report = evaluate(bands, gold)

    assert report["candidate_metrics"] == {
        "accepted": 3,
        "tp": 2,
        "fp": 1,
        "precision": pytest.approx(2 / 3),
        "recall": pytest.approx(2 / 3),
    }
    assert report["candidate_errors"] == [
        {
            "item_id": "c",
            "band": "blind_dual_exact_candidate",
            "gold_usable": False,
        }
    ]


def test_duplicate_manual_judgments_are_rejected(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text('{"item_id":"a","decision":"accept"}\n')
    second.write_text('{"item_id":"a","decision":"reject"}\n')

    with pytest.raises(ValueError, match="duplicate manual judgment"):
        load_gold([first, second])
