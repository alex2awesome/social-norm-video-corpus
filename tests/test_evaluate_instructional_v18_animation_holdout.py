import json
from pathlib import Path

import pytest

from scripts.evaluate_instructional_v18_animation_holdout import evaluate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_tsv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    path.write_text(
        "\t".join(fields)
        + "\n"
        + "".join("\t".join(str(row[field]) for field in fields) + "\n" for row in rows)
    )


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
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
            },
            {
                "audit_index": 1,
                "candidate_id": "c1",
                "item_id": "i1",
                "uid": "u1",
                "band": "qwen_scene_reject_control",
            },
        ],
    )
    write_tsv(
        visual,
        [
            {
                "audit_index": 0,
                "visual_demo": "Y",
                "visual_conclusive": "yes",
                "visual_form": "animation",
                "blind_visual_note": "event",
            },
            {
                "audit_index": 1,
                "visual_demo": "N",
                "visual_conclusive": "yes",
                "visual_form": "lecture",
                "blind_visual_note": "talk",
            },
        ],
    )
    write_tsv(
        semantic,
        [
            {
                "audit_index": 0,
                "exact_social_norm": "Y",
                "relabel_usable": "Y",
                "failure_mode": "pass",
                "corrected_event": "event",
                "semantic_note": "exact",
            },
            {
                "audit_index": 1,
                "exact_social_norm": "N",
                "relabel_usable": "N",
                "failure_mode": "visual_non_demo",
                "corrected_event": "",
                "semantic_note": "no event",
            },
        ],
    )
    return selection, visual, semantic


def test_evaluate_joins_and_scores_bands(tmp_path: Path) -> None:
    selection, visual, semantic = fixture(tmp_path)
    joined, summary = evaluate(selection, visual, semantic)
    assert joined[0]["manual_exact_social_norm"] is True
    assert joined[1]["manual_visual_demo"] is False
    assert summary["candidate_metrics"]["visual_demo"]["rate"] == 1.0
    assert summary["control_metrics_by_band"]["qwen_scene_reject_control"][
        "visual_demo"
    ]["rate"] == 0.0
    assert summary["preregistered_pass"] is False


def test_rejects_relabeling_a_visual_non_demo(tmp_path: Path) -> None:
    selection, visual, semantic = fixture(tmp_path)
    text = semantic.read_text().replace(
        "N\tN\tvisual_non_demo", "N\tY\tvisual_non_demo"
    )
    semantic.write_text(text)
    with pytest.raises(ValueError, match="visual non-demo"):
        evaluate(selection, visual, semantic)
