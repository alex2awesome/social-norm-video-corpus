import csv
import json
from pathlib import Path

import pytest

from scripts.evaluate_instructional_v14_response_holdout import evaluate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def artifacts(tmp_path: Path) -> tuple[Path, Path, Path]:
    selection = tmp_path / "selection.jsonl"
    visual = tmp_path / "visual.tsv"
    semantic = tmp_path / "semantic.tsv"
    bands = [
        "primary_v14_candidate",
        "primary_v14_candidate",
        "primary_v14_candidate",
        "response_reject",
        "v10a_reject",
    ]
    write_jsonl(
        selection,
        [
            {
                "audit_index": index,
                "candidate_id": f"c{index}",
                "item_id": f"i{index}",
                "uid": f"u{index}",
                "band": band,
            }
            for index, band in enumerate(bands)
        ],
    )
    write_tsv(
        visual,
        [
            {
                "audit_index": index,
                "candidate_id": f"c{index}",
                "visual_demo": value,
                "visual_conclusive": "Y",
                "visual_form": "scene" if value == "Y" else "talking_head",
                "visual_note": "manual",
                "protocol_exception": "N",
            }
            for index, value in enumerate(["Y", "N", "N", "Y", "N"])
        ],
    )
    write_tsv(
        semantic,
        [
            {
                "audit_index": index,
                "exact_original": "Y" if value == "Y" else "N",
                "usable_after_relabel": "Y" if value == "Y" else "N",
                "visible_polarity": (
                    "violation" if value == "Y" else "described_only"
                ),
                "failure_mechanism": (
                    "none" if value == "Y" else "reported_not_enacted"
                ),
                "corrected_event": "event" if value == "Y" else "",
                "manual_notes": "manual",
            }
            for index, value in enumerate(["Y", "N", "N", "Y", "N"])
        ],
    )
    return selection, visual, semantic


def test_evaluate_reports_bands_and_repeated_candidate_failures(
    tmp_path: Path,
) -> None:
    selection, visual, semantic = artifacts(tmp_path)
    joined, summary = evaluate(selection, visual, semantic)
    assert len(joined) == 5
    assert summary["candidate_metrics"]["visual_demo"]["positive"] == 1
    assert (
        summary["candidate_visual_false_positive_mechanisms"][
            "reported_not_enacted"
        ]
        == 2
    )
    assert (
        summary["control_metrics_by_rejection_band"]["response_reject"][
            "false_reject_visual_count"
        ]
        == 1
    )
    assert (
        summary["preregistered_checks"][
            "no_repeated_visual_false_positive_mechanism_over_5_percent"
        ]
        is False
    )
    assert summary["preregistered_pass"] is False


def test_evaluate_rejects_usable_no_demo(tmp_path: Path) -> None:
    selection, visual, semantic = artifacts(tmp_path)
    rows = list(csv.DictReader(semantic.open(), delimiter="\t"))
    rows[1]["usable_after_relabel"] = "Y"
    write_tsv(semantic, rows)
    with pytest.raises(ValueError, match="no-demo item"):
        evaluate(selection, visual, semantic)


def test_evaluate_rejects_inconclusive_visual(tmp_path: Path) -> None:
    selection, visual, semantic = artifacts(tmp_path)
    rows = list(csv.DictReader(visual.open(), delimiter="\t"))
    rows[0]["visual_conclusive"] = "maybe"
    write_tsv(visual, rows)
    with pytest.raises(ValueError, match="invalid visual judgment"):
        evaluate(selection, visual, semantic)
