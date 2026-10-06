import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_instructional_v10_prospective import evaluate


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(rows)


def _artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    selection = tmp_path / "selection.jsonl"
    visual = tmp_path / "visual.tsv"
    semantic = tmp_path / "semantic.tsv"
    _write_jsonl(
        selection,
        [
            {
                "audit_index": 0,
                "item_id": "i0",
                "uid": "u0",
                "band": "primary_candidate",
                "candidate_stage": "candidate",
                "category": "c",
                "norm": "sharing",
                "polarity": "violation",
            },
            {
                "audit_index": 1,
                "item_id": "i1",
                "uid": "u1",
                "band": "primary_candidate",
                "candidate_stage": "candidate",
                "category": "c",
                "norm": "respect",
                "polarity": "violation",
            },
            {
                "audit_index": 2,
                "item_id": "i2",
                "uid": "u2",
                "band": "control_reject",
                "candidate_stage": "v9a_reject",
                "category": "c",
                "norm": "waiting",
                "polarity": "violation",
            },
        ],
    )
    _write_tsv(
        visual,
        [
            {
                "audit_index": 0,
                "visual_demo": "D",
                "literal_description": "sharing refusal",
                "review_basis": "storyboard",
                "blind_status": "fully_blind",
            },
            {
                "audit_index": 1,
                "visual_demo": "N",
                "literal_description": "presenter",
                "review_basis": "storyboard",
                "blind_status": "fully_blind",
            },
            {
                "audit_index": 2,
                "visual_demo": "D",
                "literal_description": "child demands cake",
                "review_basis": "storyboard",
                "blind_status": "fully_blind",
            },
        ],
    )
    _write_tsv(
        semantic,
        [
            {
                "audit_index": 0,
                "exact_original": "Y",
                "usable_after_relabel": "Y",
                "visible_polarity": "violation",
                "failure_mechanism": "none",
                "semantic_notes": "exact",
            },
            {
                "audit_index": 1,
                "exact_original": "N",
                "usable_after_relabel": "N",
                "visible_polarity": "none",
                "failure_mechanism": "no_visual_demo",
                "semantic_notes": "presenter only",
            },
            {
                "audit_index": 2,
                "exact_original": "Y",
                "usable_after_relabel": "Y",
                "visible_polarity": "violation",
                "failure_mechanism": "none",
                "semantic_notes": "false reject",
            },
        ],
    )
    return selection, visual, semantic


def test_evaluate_reports_candidate_precision_and_control_recovery(
    tmp_path: Path,
) -> None:
    selection, visual, semantic = _artifacts(tmp_path)
    joined, summary = evaluate(selection, visual, semantic)

    assert len(joined) == 3
    assert summary["candidate_metrics"]["visual_demo"]["rate"] == 0.5
    assert summary["candidate_metrics"]["exact_original"]["rate"] == 0.5
    assert summary["control_metrics"]["visual_demo"]["rate"] == 1.0
    assert summary["control_metrics"]["exact_original"]["rate"] == 1.0
    assert summary["preregistered_pass"] is False
    assert summary["promotion_eligible"] is False


def test_evaluate_rejects_usable_no_demo_row(tmp_path: Path) -> None:
    selection, visual, semantic = _artifacts(tmp_path)
    rows = list(csv.DictReader(semantic.open(), delimiter="\t"))
    rows[1]["usable_after_relabel"] = "Y"
    _write_tsv(semantic, rows)

    with pytest.raises(ValueError, match="no-demo item"):
        evaluate(selection, visual, semantic)
