import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_instructional_v12_holdout import evaluate


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
                "band": "primary_v12_candidate",
            },
            {
                "audit_index": 1,
                "item_id": "i1",
                "uid": "u1",
                "band": "primary_v12_candidate",
            },
            {
                "audit_index": 2,
                "item_id": "i2",
                "uid": "u2",
                "band": "control_qwen_exact_v12_reject",
            },
        ],
    )
    _write_tsv(
        visual,
        [
            {
                "audit_index": 0,
                "visual_demo": "Y",
                "visual_form": "enacted_live",
                "protocol_exception": "true",
                "visual_note": "event",
            },
            {
                "audit_index": 1,
                "visual_demo": "N",
                "visual_form": "talking_head",
                "protocol_exception": "false",
                "visual_note": "presenter",
            },
            {
                "audit_index": 2,
                "visual_demo": "Y",
                "visual_form": "animated_scene",
                "protocol_exception": "false",
                "visual_note": "event",
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
                "corrected_event": "event",
                "manual_notes": "exact",
            },
            {
                "audit_index": 1,
                "exact_original": "N",
                "usable_after_relabel": "N",
                "visible_polarity": "described_only",
                "failure_mechanism": "no_performed_event",
                "corrected_event": "none",
                "manual_notes": "no demo",
            },
            {
                "audit_index": 2,
                "exact_original": "Y",
                "usable_after_relabel": "Y",
                "visible_polarity": "violation",
                "failure_mechanism": "none",
                "corrected_event": "event",
                "manual_notes": "false reject",
            },
        ],
    )
    return selection, visual, semantic


def test_evaluate_reports_v12_candidate_control_and_blind_metrics(
    tmp_path: Path,
) -> None:
    selection, visual, semantic = _artifacts(tmp_path)
    joined, summary = evaluate(selection, visual, semantic)
    assert len(joined) == 3
    assert summary["candidate_metrics"]["visual_demo"]["rate"] == 0.5
    assert summary["candidate_metrics"]["fully_blind_visual_demo"]["rate"] == 0.0
    assert summary["control_metrics"]["exact_original"]["rate"] == 1.0
    assert summary["coverage"]["protocol_exceptions"] == 1
    assert summary["preregistered_pass"] is False
    assert summary["promotion_eligible"] is False


def test_evaluate_rejects_relabel_usable_no_demo(tmp_path: Path) -> None:
    selection, visual, semantic = _artifacts(tmp_path)
    rows = list(csv.DictReader(semantic.open(), delimiter="\t"))
    rows[1]["usable_after_relabel"] = "Y"
    _write_tsv(semantic, rows)
    with pytest.raises(ValueError, match="no-demo item"):
        evaluate(selection, visual, semantic)


def test_evaluate_rejects_unresolved_visual_pass(tmp_path: Path) -> None:
    selection, visual, semantic = _artifacts(tmp_path)
    rows = list(csv.DictReader(visual.open(), delimiter="\t"))
    rows[0]["visual_demo"] = "U"
    _write_tsv(visual, rows)
    with pytest.raises(ValueError, match="exact item must be a visible"):
        evaluate(selection, visual, semantic)
