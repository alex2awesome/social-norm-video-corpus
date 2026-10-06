import json

import pytest

from scripts.labeling_functions_v1 import make_lf_record
from scripts.lf_matrix_v1 import (
    build_matrices,
    load_gold,
    matrix_diagnostics,
    wilson_lower,
)


def rec(item, lf, vote, family="event_action", target="demonstration_present"):
    return make_lf_record(
        item_id=item,
        pillar="instructional",
        target=target,
        lf_id=lf,
        family=family,
        vote=vote,
        abstain_reason="no evidence" if vote == 0 else None,
    )


def test_matrices_group_by_pillar_and_target():
    records = [
        rec("a", "lf1", 1),
        rec("a", "lf2", -1, family="visual_depiction"),
        rec("b", "lf1", 0),
        rec("a", "lf3", 1, target="label_alignment", family="explicit_semantics"),
    ]
    matrices = build_matrices(records)
    assert set(matrices) == {
        ("instructional", "demonstration_present"),
        ("instructional", "label_alignment"),
    }
    demo = matrices[("instructional", "demonstration_present")]
    assert demo["items"] == {"a", "b"}
    assert set(demo["lfs"]) == {"lf1", "lf2"}


def test_conflicting_duplicate_votes_are_an_error():
    with pytest.raises(ValueError, match="duplicate"):
        build_matrices([rec("a", "lf1", 1), rec("a", "lf1", -1)])
    # Identical duplicates are tolerated (idempotent re-emission).
    build_matrices([rec("a", "lf1", 1), rec("a", "lf1", 1)])


def test_diagnostics_coverage_conflict_and_correlated_family_flag():
    records = []
    for i in range(10):
        item = f"item{i}"
        vote = 1 if i < 5 else -1
        records.append(rec(item, "lf_a", vote))
        records.append(rec(item, "lf_b", vote))  # perfectly correlated, same family
        records.append(rec(item, "lf_c", -vote, family="visual_depiction"))
    matrix = build_matrices(records)[("instructional", "demonstration_present")]
    diag = matrix_diagnostics(matrix)
    assert diag["items"] == 10 and diag["lfs"] == 3
    per_lf = {row["lf_id"]: row for row in diag["per_lf"]}
    assert per_lf["lf_a"]["coverage"] == 1.0
    flagged = {
        (p["lf_a"], p["lf_b"]) for p in diag["correlated_same_family_pairs"]
    }
    assert ("lf_a", "lf_b") in flagged
    ab = next(p for p in diag["pairwise"] if (p["lf_a"], p["lf_b"]) == ("lf_a", "lf_b"))
    assert ab["conflict_rate"] == 0.0 and ab["correlation"] == pytest.approx(1.0)
    ac = next(p for p in diag["pairwise"] if (p["lf_a"], p["lf_b"]) == ("lf_a", "lf_c"))
    assert ac["conflict_rate"] == 1.0


def test_gold_precision_recall_and_wilson():
    records = [rec(f"item{i}", "lf_a", 1 if i < 6 else 0) for i in range(8)]
    matrix = build_matrices(records)[("instructional", "demonstration_present")]
    gold = {f"item{i}": (1 if i in {0, 1, 2, 6} else -1) for i in range(8)}
    diag = matrix_diagnostics(matrix, gold)
    row = diag["per_lf"][0]
    assert row["gold_tp"] == 3 and row["gold_fp"] == 3 and row["gold_fn"] == 1
    assert row["gold_precision"] == pytest.approx(0.5)
    assert row["gold_recall"] == pytest.approx(0.75)
    assert 0 < row["gold_precision_wilson_lower"] < 0.5


def test_wilson_lower_bounds():
    assert wilson_lower(0, 0) is None
    assert wilson_lower(10, 10) < 1.0
    assert wilson_lower(0, 10) == 0.0


def test_load_gold_filters_and_validates(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"item_id": "a", "pillar": "instructional", "target": "demonstration_present", "label": 1},
                {"item_id": "b", "pillar": "witnessed", "target": "reaction_grounded", "label": -1},
            ]
        )
    )
    gold = load_gold(path, "instructional", "demonstration_present")
    assert gold == {"a": 1}
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"item_id": "a", "pillar": "instructional", "target": "demonstration_present", "label": 0}))
    with pytest.raises(ValueError, match="gold label"):
        load_gold(bad, "instructional", "demonstration_present")
