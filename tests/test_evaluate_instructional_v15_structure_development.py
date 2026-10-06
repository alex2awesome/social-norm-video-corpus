import json
from pathlib import Path

from scripts.evaluate_instructional_v15_structure_development import evaluate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_evaluate_reports_retained_candidate_false_positive(tmp_path: Path):
    manual_path = tmp_path / "manual.jsonl"
    gemma_path = tmp_path / "gemma.jsonl"
    manual = [
        {
            "audit_index": 0,
            "candidate_id": "c0",
            "is_v14_candidate": True,
            "manual_visual_demo": True,
            "manual_usable_after_relabel": True,
            "manual_exact_original": True,
            "manual_failure_mechanism": "none",
        },
        {
            "audit_index": 1,
            "candidate_id": "c1",
            "is_v14_candidate": True,
            "manual_visual_demo": False,
            "manual_usable_after_relabel": False,
            "manual_exact_original": False,
            "manual_failure_mechanism": "reported_not_enacted",
        },
        {
            "audit_index": 2,
            "candidate_id": "c2",
            "is_v14_candidate": False,
            "manual_visual_demo": True,
            "manual_usable_after_relabel": True,
            "manual_exact_original": False,
            "manual_failure_mechanism": "norm_mismatch",
        },
    ]
    gemma = [
        {
            "audit_index": index,
            "candidate_id": f"c{index}",
            "error": None,
            "result": {
                "structural_demo_pass": "yes",
                "interaction_structure": "specific_action_with_target",
                "evidence": "event",
            },
        }
        for index in range(3)
    ]
    write_jsonl(manual_path, manual)
    write_jsonl(gemma_path, gemma)
    result = evaluate(gemma_path, manual_path)
    assert result["gemma_vs_manual"]["visual_demo"]["precision"] == 2 / 3
    assert result["candidate_false_positive_retention"]["retained_by_gemma"] == [
        1
    ]
    assert result["v14_candidate_and_gemma_pass"]["selected"] == 2
