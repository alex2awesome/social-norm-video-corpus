import csv
import json
from pathlib import Path

from scripts.discover_instructional_vlm_ensembles import discover


def test_discover_scores_boolean_ensembles(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    semantic = tmp_path / "semantic.tsv"
    gemma = tmp_path / "gemma.jsonl"
    rows = [
        {
            "audit_index": 0,
            "item_id": "i0",
            "qwen_v11": {"result": {"exact_violation_demo": "yes"}},
            "glm_v11": {"result": {"exact_violation_demo": "no"}},
        },
        {
            "audit_index": 1,
            "item_id": "i1",
            "qwen_v11": {"result": {"exact_violation_demo": "yes"}},
            "glm_v11": {"result": {"exact_violation_demo": "yes"}},
        },
    ]
    selection.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with semantic.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["audit_index", "exact_original"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(
            [
                {"audit_index": 0, "exact_original": "Y"},
                {"audit_index": 1, "exact_original": "N"},
            ]
        )
    gemma.write_text(
        "".join(
            json.dumps(
                {
                    "item_id": item_id,
                    "error": None,
                    "result": {"exact_violation_demo": value},
                }
            )
            + "\n"
            for item_id, value in (("i0", "yes"), ("i1", "no"))
        )
    )
    summary = discover(selection, semantic, gemma)
    assert summary["rules"]["qwen"]["precision"] == 0.5
    assert summary["rules"]["qwen_and_gemma"]["precision"] == 1.0
    assert summary["rules"]["qwen_and_gemma"]["recall"] == 1.0
