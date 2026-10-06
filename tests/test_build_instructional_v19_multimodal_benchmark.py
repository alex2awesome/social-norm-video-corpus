import json
from pathlib import Path

from scripts.build_instructional_v19_multimodal_benchmark import build_cohort


def test_build_cohort_supports_v18_field_names(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    visual = tmp_path / "visual.tsv"
    semantic = tmp_path / "semantic.tsv"
    selection.write_text(
        json.dumps(
            {
                "audit_index": 0,
                "item_id": "i",
                "uid": "u",
                "source_clip": "/remote/demo.mp4",
                "source_clip_sha256": "abc",
            }
        )
        + "\n"
    )
    visual.write_text(
        "audit_index\tvisual_demo\n"
        "0\tY\n"
    )
    semantic.write_text(
        "audit_index\texact_social_norm\trelabel_usable\tfailure_mode\n"
        "0\tN\tY\tlabel_mismatch\n"
    )
    rows = build_cohort("v18", selection, visual, semantic)
    assert rows == [
        {
            "item_id": "i",
            "uid": "u",
            "pillar": "instructional",
            "source_clip": "/remote/demo.mp4",
            "source_sha256": "abc",
            "audit_cohort": "v18",
            "audit_index": 0,
            "gold_scene_visible": True,
            "gold_usable": True,
            "gold_exact_social_norm": False,
            "gold_failure_mode": "label_mismatch",
        }
    ]
