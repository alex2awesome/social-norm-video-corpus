import json
from pathlib import Path

from scripts.evaluate_instructional_v15_utterance_anchor_development import score


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_score_reports_retained_precision_and_recall_cost(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    evaluated = tmp_path / "evaluated.jsonl"
    write_jsonl(
        selection,
        [
            {
                "audit_index": 0,
                "candidate_id": "c0",
                "item_id": "i0",
                "uid": "u0",
                "band": "primary_v14_candidate",
                "v9a": {
                    "result": {
                        "evidence_source": "physical_action",
                        "literal_action_or_situated_utterance": "pushes person",
                    }
                },
            },
            {
                "audit_index": 1,
                "candidate_id": "c1",
                "item_id": "i1",
                "uid": "u1",
                "band": "primary_v14_candidate",
                "v9a": {
                    "result": {
                        "evidence_source": "situated_dialogue_or_subtitles",
                        "literal_action_or_situated_utterance": "person speaks",
                    }
                },
            },
            {
                "audit_index": 2,
                "candidate_id": "c2",
                "item_id": "i2",
                "uid": "u2",
                "band": "response_reject",
                "v9a": {"result": {"evidence_source": "physical_action"}},
            },
        ],
    )
    write_jsonl(
        evaluated,
        [
            {
                "audit_index": 0,
                "item_id": "i0",
                "manual_visual_demo": True,
                "manual_usable_after_relabel": True,
                "manual_exact_original": True,
                "manual_failure_mechanism": "none",
            },
            {
                "audit_index": 1,
                "item_id": "i1",
                "manual_visual_demo": False,
                "manual_usable_after_relabel": False,
                "manual_exact_original": False,
                "manual_failure_mechanism": "reported_not_enacted",
            },
            {
                "audit_index": 2,
                "item_id": "i2",
                "manual_visual_demo": True,
                "manual_usable_after_relabel": True,
                "manual_exact_original": True,
                "manual_failure_mechanism": "none",
            },
        ],
    )
    rows, summary = score(selection, evaluated)
    assert len(rows) == 2
    assert summary["coverage"] == {
        "v14_candidates": 2,
        "retained": 1,
        "rejected": 1,
        "retained_uids": 1,
    }
    assert summary["retained_metrics"]["visual_demo"]["rate"] == 1.0
    assert summary["rejected_indices"] == [1]
    assert summary["promotion_eligible"] is False

