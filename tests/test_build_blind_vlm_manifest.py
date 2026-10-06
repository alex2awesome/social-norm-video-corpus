import json
from pathlib import Path

import pytest

from scripts.build_blind_vlm_manifest import build


def test_builds_runner_rows_without_semantics(tmp_path: Path) -> None:
    source = tmp_path / "blind.jsonl"
    source.write_text(
        json.dumps(
            {
                "audit_index": 0,
                "candidate_id": "opaque-0",
                "sheet_path": "/sealed/opaque-0.jpg",
                "sheet_sha256": "a" * 64,
            }
        )
        + "\n"
    )
    row = build(source, 36)[0]
    assert row["item_id"] == "opaque-0"
    assert row["frame_count"] == 36
    assert "norm" not in row


def test_rejects_semantic_field(tmp_path: Path) -> None:
    source = tmp_path / "blind.jsonl"
    source.write_text(
        json.dumps(
            {
                "audit_index": 0,
                "candidate_id": "opaque-0",
                "sheet_path": "/sealed/opaque-0.jpg",
                "sheet_sha256": "a" * 64,
                "norm": "leak",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="semantic"):
        build(source, 36)
