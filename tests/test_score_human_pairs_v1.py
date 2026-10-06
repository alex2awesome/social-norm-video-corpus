import pytest

from experiments.score_human_pairs_v1 import score


def pairs():
    return [
        {"pair_id": "p1", "positive_side": "A", "vlm_choice_correct": True},
        {"pair_id": "p2", "positive_side": "B", "vlm_choice_correct": False},
        {"pair_id": "p3", "positive_side": "A", "vlm_choice_correct": None},
        {"pair_id": "p4", "positive_side": "B", "vlm_choice_correct": None},
    ]


def test_scoring_accuracy_confidence_flags_and_headtohead():
    judgments = [
        {"pair_id": "p1", "choice": "A", "confidence": "sure", "flags": []},
        {"pair_id": "p2", "choice": "A", "confidence": "guess", "flags": ["both_calm"]},
        {"pair_id": "p3", "choice": "cant_tell", "confidence": None, "flags": ["same_moment"]},
        {"pair_id": "p4", "choice": "B", "confidence": "sure", "flags": []},
    ]
    report = score(pairs(), judgments)
    assert report["decided"] == 3 and report["cant_tell"] == 1
    assert report["human_accuracy"] == pytest.approx(2 / 3)
    assert report["sure_n"] == 2 and report["sure_accuracy"] == 1.0
    assert report["flag_counts"] == {"both_calm": 1, "same_moment": 1}
    h2h = report["vlm_headtohead"]
    assert h2h["pairs_both_saw"] == 2
    assert h2h["human_decided_on_those"] == 2
    assert h2h["human_correct"] == 1 and h2h["vlm_correct"] == 1


def test_unknown_pair_rejected():
    with pytest.raises(ValueError, match="unknown"):
        score(pairs(), [{"pair_id": "zzz", "choice": "A"}])
