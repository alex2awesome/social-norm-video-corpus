from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_instructional_v13_holdout import evaluate


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def make_fixture(tmp_path: Path, count: int = 30) -> tuple[Path, Path, Path]:
    selection = tmp_path / "selection.jsonl"
    visual = tmp_path / "visual.tsv"
    semantic = tmp_path / "semantic.tsv"
    write_jsonl(
        selection,
        [
            {
                "audit_index": i,
                "item_id": f"item-{i}",
                "uid": f"uid-{i}",
                "band": "primary_v13_candidate",
                "category": "instr_test",
                "norm": "respect",
                "polarity": "violation",
            }
            for i in range(count)
        ],
    )
    write_tsv(
        visual,
        [
            {
                "audit_index": str(i),
                "visual_demo": "Y",
                "visual_form": "live_roleplay",
                "visual_note": "visible event",
                "protocol_exception": "false",
            }
            for i in range(count)
        ],
    )
    write_tsv(
        semantic,
        [
            {
                "audit_index": str(i),
                "exact_original": "Y",
                "usable_after_relabel": "Y",
                "visible_polarity": "violation",
                "failure_mechanism": "none",
                "corrected_event": "visible event",
                "manual_notes": "aligned",
            }
            for i in range(count)
        ],
    )
    return selection, visual, semantic


def test_perfect_fixture_passes_as_shadow_only(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path)
    joined, summary = evaluate(*paths)
    assert len(joined) == 30
    assert summary["preregistered_pass"] is True
    assert summary["promotion_eligible"] is False


def test_visual_false_positive_fails_precision_gate(tmp_path: Path) -> None:
    selection, visual, semantic = make_fixture(tmp_path)
    visual_rows = list(csv.DictReader(visual.open(), delimiter="\t"))
    semantic_rows = list(csv.DictReader(semantic.open(), delimiter="\t"))
    for i in (0, 1):
        visual_rows[i]["visual_demo"] = "N"
        visual_rows[i]["visual_form"] = "talking_head"
        semantic_rows[i].update(
            {
                "exact_original": "N",
                "usable_after_relabel": "N",
                "visible_polarity": "described_only",
                "failure_mechanism": "no_performed_event",
            }
        )
    write_tsv(visual, visual_rows)
    write_tsv(semantic, semantic_rows)
    _, summary = evaluate(selection, visual, semantic)
    assert summary["candidate_metrics"]["visual_demo"]["rate"] == pytest.approx(
        28 / 30
    )
    assert summary["preregistered_pass"] is False


def test_non_demo_cannot_be_relabel_usable(tmp_path: Path) -> None:
    selection, visual, semantic = make_fixture(tmp_path)
    visual_rows = list(csv.DictReader(visual.open(), delimiter="\t"))
    semantic_rows = list(csv.DictReader(semantic.open(), delimiter="\t"))
    visual_rows[0]["visual_demo"] = "N"
    semantic_rows[0].update(
        {
            "exact_original": "N",
            "usable_after_relabel": "Y",
            "failure_mechanism": "no_performed_event",
        }
    )
    write_tsv(visual, visual_rows)
    write_tsv(semantic, semantic_rows)
    with pytest.raises(ValueError, match="non-demo item"):
        evaluate(selection, visual, semantic)
