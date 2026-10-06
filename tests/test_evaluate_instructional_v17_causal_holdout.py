import csv
import json
from pathlib import Path

from scripts.evaluate_instructional_v17_causal_holdout import evaluate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_v17_reports_exact_gate_and_all_control_bands(tmp_path: Path) -> None:
    bands = [
        "candidate",
        "candidate",
        "glm_v10a_reject",
        "qwen_v16_reject",
        "gemma_causal_reject",
    ]
    selection = tmp_path / "selection.jsonl"
    visual = tmp_path / "visual.tsv"
    semantic = tmp_path / "semantic.tsv"
    write_jsonl(
        selection,
        [
            {
                "audit_index": i,
                "candidate_id": f"c{i}",
                "item_id": f"i{i}",
                "uid": f"u{i}",
                "band": band,
            }
            for i, band in enumerate(bands)
        ],
    )
    judgments = ["Y", "Y", "N", "Y", "N"]
    exact = ["Y", "N", "N", "Y", "N"]
    write_tsv(
        visual,
        [
            {
                "audit_index": i,
                "visual_demo": value,
                "visual_conclusive": "Y",
                "visual_form": "scene" if value == "Y" else "talking_head",
                "visual_note": "manual",
                "protocol_exception": "N",
            }
            for i, value in enumerate(judgments)
        ],
    )
    write_tsv(
        semantic,
        [
            {
                "audit_index": i,
                "exact_original": exact_value,
                "usable_after_relabel": visual_value,
                "visible_polarity": (
                    "violation" if visual_value == "Y" else "described_only"
                ),
                "failure_mechanism": (
                    "none"
                    if exact_value == "Y"
                    else (
                        "label_mismatch"
                        if visual_value == "Y"
                        else "reported_not_enacted"
                    )
                ),
                "corrected_event": "",
                "manual_notes": "manual",
            }
            for i, (visual_value, exact_value) in enumerate(
                zip(judgments, exact)
            )
        ],
    )

    joined, summary = evaluate(selection, visual, semantic)
    assert len(joined) == 5
    assert summary["candidate_metrics"]["visual_demo"]["positive"] == 2
    assert summary["candidate_metrics"]["exact_original"]["positive"] == 1
    assert (
        summary["preregistered_checks"][
            "candidate_exact_original_at_least_90_percent"
        ]
        is False
    )
    assert set(summary["control_metrics_by_rejection_band"]) == {
        "glm_v10a_reject",
        "qwen_v16_reject",
        "gemma_causal_reject",
    }


def test_v17_preserves_uncertain_visual_judgment_and_fails_closed(
    tmp_path: Path,
) -> None:
    selection = tmp_path / "selection.jsonl"
    visual = tmp_path / "visual.tsv"
    semantic = tmp_path / "semantic.tsv"
    write_jsonl(
        selection,
        [
            {
                "audit_index": 0,
                "candidate_id": "c0",
                "item_id": "i0",
                "uid": "u0",
                "band": "candidate",
            }
        ],
    )
    write_tsv(
        visual,
        [
            {
                "audit_index": 0,
                "visual_demo": "U",
                "visual_conclusive": "no",
                "visual_form": "ambiguous_cctv",
                "blind_visual_note": "The action cannot be resolved.",
                "protocol_exception": "N",
            }
        ],
    )
    write_tsv(
        semantic,
        [
            {
                "audit_index": 0,
                "exact_original": "N",
                "usable_after_relabel": "N",
                "visible_polarity": "unclear",
                "failure_mechanism": "event_ambiguous",
                "corrected_event": "",
                "manual_notes": "Follow-up frames remained ambiguous.",
            }
        ],
    )

    joined, summary = evaluate(selection, visual, semantic)
    assert joined[0]["manual_visual_demo"] is False
    assert joined[0]["manual_visual_demo_label"] == "U"
    assert joined[0]["manual_visual_uncertain"] is True
    assert joined[0]["manual_visual_conclusive"] is False
    assert joined[0]["manual_visual_note"] == "The action cannot be resolved."
    assert summary["candidate_metrics"]["visual_demo"]["positive"] == 0
    assert (
        summary["preregistered_checks"][
            "all_candidates_and_controls_conclusive"
        ]
        is False
    )
