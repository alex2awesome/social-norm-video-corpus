import json
from pathlib import Path

from scripts.export_witnessed_atomic_holdout import build_rows


def test_build_rows_preserves_order_and_strict_gold(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "visual_samples": [
                    {
                        "audit_index": 0,
                        "uid": "source__a",
                        "norm": "courtesy",
                        "reaction": "stop",
                        "reaction_context": "please stop",
                    },
                    {
                        "audit_index": 1,
                        "uid": "source__b",
                        "norm": "privacy",
                        "reaction": "leave",
                        "reaction_context": "please leave",
                    },
                ]
            }
        )
    )
    (tmp_path / "manual_blind_witnessed_review.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"audit_index": 0, "action_visibility": "yes"}),
                json.dumps({"audit_index": 1, "action_visibility": "needs_motion"}),
            ]
        )
        + "\n"
    )
    (tmp_path / "manual_revealed_label_adjudication.json").write_text(
        json.dumps(
            {
                "strict_witnessed_contract": {
                    "accepted_indices": [0],
                    "source_authenticity_unresolved_indices": [1],
                }
            }
        )
    )
    (tmp_path / "all_clip_transcripts_tiny_en.json").write_text(
        json.dumps(
            {
                "records": [
                    {"uid": "0_source__a", "segments": [{"text": "a"}]},
                    {"uid": "1_source__b", "segments": [{"text": "b"}]},
                ]
            }
        )
    )

    rows = build_rows(tmp_path)

    assert [row["uid"] for row in rows] == ["source__a", "source__b"]
    assert rows[0]["gold_usable"] is True
    assert rows[1]["gold_usable"] is False
    assert rows[1]["gold_source_authenticity_unresolved"] is True
    assert rows[0]["source_clip"] == "clips/0_source__a.mp4"
    assert rows[0]["proxy_clip"] == "proxy_clips_1fps/0_source__a.mp4"
