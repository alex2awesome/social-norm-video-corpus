import json
from pathlib import Path

from scripts.shadow_score_monitor import render_progress, result_progress


def write_rows(path: Path, rows: list[dict]):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_monitor_counts_latest_success_and_preserves_attempts(tmp_path: Path):
    path = tmp_path / "scores.jsonl"
    write_rows(
        path,
        [
            {"item_id": "a", "result": None, "error": "timeout"},
            {"item_id": "a", "result": {"ok": True}, "error": None},
            {"item_id": "b", "result": None, "error": "bad"},
        ],
    )
    assert result_progress(path) == {
        "attempts": 3,
        "unique_items": 2,
        "successful": 1,
        "latest_failures": 1,
    }


def test_render_progress_uses_proxy_clip_as_success(tmp_path: Path):
    path = tmp_path / "renders.jsonl"
    write_rows(
        path,
        [
            {"item_id": "a", "proxy_clip": "/tmp/a.mp4", "error": None},
            {"item_id": "b", "proxy_clip": None, "error": "bad"},
        ],
    )
    assert render_progress(path)["successful"] == 1
