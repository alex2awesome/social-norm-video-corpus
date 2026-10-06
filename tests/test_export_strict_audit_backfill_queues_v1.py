import json
from pathlib import Path

import pytest

from scripts.export_strict_audit_backfill_queues_v1 import export


def test_exports_only_missing_layers_and_bounds_commentary(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    rows = [
        {
            "item_id": "instructional:u:0",
            "pillar": "instructional",
            "uid": "u",
            "media_path": "data/instructional/u/demo.mp4",
            "media_present": True,
            "low_level_visual_complete": False,
            "strict_audit_needed": True,
            "start_sec": 100,
            "end_sec": 120,
            "norm": "n",
        },
        {
            "item_id": "commentary:c:0",
            "pillar": "commentary",
            "uid": "c",
            "media_path": "data/discussion_video/c.mp4",
            "media_present": True,
            "low_level_visual_complete": False,
            "strict_audit_needed": True,
            "start_sec": 5,
            "end_sec": 8,
            "norm": "n",
        },
        {
            "item_id": "witnessed:w:0",
            "pillar": "witnessed",
            "uid": "w",
            "media_path": None,
            "media_present": False,
            "low_level_visual_complete": False,
            "strict_audit_needed": True,
        },
    ]
    manifest = tmp_path / "labels.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    out = tmp_path / "queues"
    summary = export(manifest, root, out)
    assert summary["low_level_delta_by_pillar"] == {
        "commentary": 1,
        "instructional": 1,
    }
    low = [json.loads(line) for line in (out / "low_level_delta_manifest.jsonl").read_text().splitlines()]
    assert low[0]["media_start_sec"] is None
    assert low[1]["media_start_sec"] == 0.0
    assert low[1]["media_end_sec"] == 20.0


def test_refuses_overwrite(tmp_path: Path) -> None:
    manifest = tmp_path / "labels.jsonl"
    manifest.write_text("")
    out = tmp_path / "queues"
    out.mkdir()
    with pytest.raises(FileExistsError):
        export(manifest, tmp_path, out)
