import csv
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "evaluate_instructional_v9_corpus_validation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_v9_corpus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def real_audit():
    audit = ROOT / "audit_runs" / "20260728_instructional_v9_corpus_validation_v1"
    return MODULE.evaluate(
        audit / "validation_selection.jsonl",
        audit / "blind" / "manual_visual_ledger.tsv",
        audit / "manual_semantic_ledger.tsv",
    )


def test_wilson_is_bounded_and_contains_rate():
    low, high = MODULE.wilson_95(92, 99)
    assert 0 <= low < 92 / 99 < high <= 1


def test_real_audit_is_complete_and_fails_frozen_promotion_rule():
    joined, summary = real_audit()
    assert len(joined) == 166
    assert summary["review_coverage"] == {
        "all_selected": 166,
        "all_selected_reviewed": True,
        "blind_before_semantic_reveal": 165,
        "recorded_blind_protocol_exceptions": 1,
    }
    assert summary["promotion_pass"] is False
    assert (
        summary["bands"]["primary_violation_all"]["visual_demo"]["positive"]
        == 92
    )
    assert summary["primary_no_demo_indices"] == [8, 30, 82, 104, 109, 121, 160]


def test_all_bands_and_source_clusters_are_reported():
    _, summary = real_audit()
    assert {
        band: metrics["visual_demo"]["n"]
        for band, metrics in summary["bands"].items()
    } == {
        "primary_violation_all": 99,
        "comparison_correct": 30,
        "comparison_explanation": 7,
        "control_violation_reject": 30,
        "all_166": 166,
    }
    clusters = summary["primary_source_clusters"]
    assert clusters["distinct_sources"] == 90
    assert clusters["exact_original_all_clips"]["n"] == 90


def test_no_demo_cannot_be_marked_semantically_usable(tmp_path):
    audit = ROOT / "audit_runs" / "20260728_instructional_v9_corpus_validation_v1"
    semantic_path = audit / "manual_semantic_ledger.tsv"
    with semantic_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
        fields = list(rows[0])
    rows[8]["usable_after_relabel"] = "Y"
    bad = tmp_path / "bad.tsv"
    with bad.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="no-demo item cannot be label-usable"):
        MODULE.evaluate(
            audit / "validation_selection.jsonl",
            audit / "blind" / "manual_visual_ledger.tsv",
            bad,
        )
