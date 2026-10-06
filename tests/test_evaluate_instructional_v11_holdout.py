import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_instructional_v11_holdout import evaluate


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
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
            },
            {
                "audit_index": 1,
                "item_id": "i1",
                "uid": "u1",
                "band": "primary_candidate",
                "candidate_stage": "candidate",
            },
            {
                "audit_index": 2,
                "item_id": "i2",
                "uid": "u2",
                "band": "control_reject",
                "candidate_stage": "glm_only_exact",
            },
        ],
    )
    _write_tsv(
        visual,
        [
            {"audit_index": 0, "visual_demo": "Y", "literal_description": "event", "review_basis": "storyboard", "blind_status": "frozen"},
            {"audit_index": 1, "visual_demo": "N", "literal_description": "presenter", "review_basis": "storyboard", "blind_status": "frozen"},
            {"audit_index": 2, "visual_demo": "Y", "literal_description": "event", "review_basis": "storyboard", "blind_status": "frozen"},
        ],
    )
    _write_tsv(
        semantic,
        [
            {"audit_index": 0, "exact_original": "Y", "usable_after_relabel": "Y", "visible_polarity": "violation", "failure_mechanism": "none", "manual_notes": "exact"},
            {"audit_index": 1, "exact_original": "N", "usable_after_relabel": "N", "visible_polarity": "none", "failure_mechanism": "no_performed_event", "manual_notes": "no demo"},
            {"audit_index": 2, "exact_original": "Y", "usable_after_relabel": "Y", "visible_polarity": "violation", "failure_mechanism": "none", "manual_notes": "false reject"},
        ],
    )
    return selection, visual, semantic


def test_evaluate_reports_candidate_and_control_metrics(tmp_path: Path) -> None:
    selection, visual, semantic = _artifacts(tmp_path)
    joined, summary = evaluate(selection, visual, semantic)
    assert len(joined) == 3
    assert summary["candidate_metrics"]["visual_demo"]["rate"] == 0.5
    assert summary["candidate_metrics"]["exact_original"]["rate"] == 0.5
    assert summary["control_metrics"]["exact_original"]["rate"] == 1.0
    assert summary["preregistered_pass"] is False
    assert summary["promotion_eligible"] is False


def test_evaluate_rejects_relabel_usable_no_demo(tmp_path: Path) -> None:
    selection, visual, semantic = _artifacts(tmp_path)
    rows = list(csv.DictReader(semantic.open(), delimiter="\t"))
    rows[1]["usable_after_relabel"] = "Y"
    _write_tsv(semantic, rows)
    with pytest.raises(ValueError, match="no-demo item"):
        evaluate(selection, visual, semantic)
