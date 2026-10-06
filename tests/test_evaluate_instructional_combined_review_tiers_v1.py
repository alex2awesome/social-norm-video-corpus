from pathlib import Path

from scripts.evaluate_instructional_combined_review_tiers_v1 import evaluate, load_cohort


ROOT = Path(__file__).resolve().parents[1]


def test_real_manual_cohorts_support_monotonic_review_tiers():
    import csv
    import json

    specs = {
        "fresh_36": (
            ROOT / "audit_runs/20260805_instructional_v22_fresh_holdout_v1/sealed_selection.jsonl",
            ROOT / "audit_runs/20260805_instructional_v22_fresh_holdout_v1/manual_blind_ledger.tsv",
        ),
        "fresh_100": (
            ROOT / "audit_runs/20260806_instructional_demo_consensus_v2_fresh_transfer/fresh_holdout_100/sealed_selection.jsonl",
            ROOT / "audit_runs/20260806_instructional_v23_replication_100/manual_blind_ledger.tsv",
        ),
    }
    cohorts = {}
    for name, (selection, ledger) in specs.items():
        selected = [json.loads(line) for line in selection.read_text().splitlines() if line]
        with ledger.open(newline="") as handle:
            labels = list(csv.DictReader(handle, delimiter="\t"))
        cohorts[name] = load_cohort(name, selected, labels)
    report = evaluate(cohorts)
    assert report["review_tiering_promoted"] is True
    assert report["combined"]["tiers"]["both"] == {
        "items": 22,
        "visual_demos": 16,
        "visual_demo_rate": 16 / 22,
    }
    assert report["combined"]["tiers"]["polarity_only"]["visual_demo_rate"] == 14 / 85
    assert report["combined"]["tiers"]["neither"]["visual_demo_rate"] == 2 / 29
    assert report["combined_both_recall"] == 0.5
    assert report["combined"]["tiers"]["retro_only"]["items"] == 0
    assert report["automatic_acceptance"] is False


def test_nonmonotonic_replication_fails_gate():
    rows = []
    for tier, positives, items in (("both", 4, 7), ("polarity_only", 5, 5), ("neither", 0, 2)):
        rows.extend({"tier": tier, "visual_demo": index < positives} for index in range(items))
    report = evaluate({"a": rows, "b": rows})
    assert report["review_tiering_promoted"] is False
    assert report["allowed_use"] == "none"
