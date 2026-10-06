from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

from scripts.backfill_commentary_videos import (
    adopt_existing,
    download_one,
    iter_records,
    stable_video,
)


def test_records_are_deterministic_and_keep_urls(tmp_path):
    discussion = tmp_path / "discussion"
    discussion.mkdir()
    for uid in ("dailymotion__x2", "dailymotion__x1"):
        (discussion / f"{uid}.json").write_text(json.dumps({
            "video_id": uid,
            "url": f"https://example.test/{uid}",
            "source": "test",
        }))

    first = list(iter_records(discussion))
    second = list(iter_records(discussion))

    assert first == second
    assert {row["uid"] for row in first} == {"dailymotion__x1", "dailymotion__x2"}
    assert all(row["url"].startswith("https://example.test/") for row in first)


def test_adopt_existing_preserves_raw_and_is_discoverable(tmp_path):
    raw = tmp_path / "raw.mp4"
    raw.write_bytes(b"source-video")
    stable = tmp_path / "stable" / "uid.mp4"

    method = adopt_existing(raw, stable)

    assert method in {"hardlink", "copy"}
    assert raw.read_bytes() == stable.read_bytes() == b"source-video"
    assert stable_video(stable.parent, "uid") == stable
    if method == "hardlink":
        assert os.stat(raw).st_ino == os.stat(stable).st_ino


def test_download_passes_explicit_auth_and_player_client(tmp_path):
    stage = tmp_path / "stage"
    stable = tmp_path / "stable"
    stage.mkdir()
    stable.mkdir()
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape HTTP Cookie File\n")
    completed = Mock(returncode=0, stdout="", stderr="")

    def fake_download(*_args, **_kwargs):
        (stage / "youtube__x.mp4").write_bytes(b"video")
        return completed

    with (
        patch("scripts.backfill_commentary_videos.subprocess.run", side_effect=fake_download) as run,
        patch("scripts.backfill_commentary_videos.probe_media", return_value={
            "duration": 1.0, "video_streams": 1, "audio_streams": 1,
        }),
        patch("scripts.backfill_commentary_videos.sha256_file", return_value="digest"),
    ):
        result = download_one(
            {"uid": "youtube__x", "url": "https://example.test/x"},
            stage, stable, "ffprobe", 720, 1800, "8M", cookies, "android",
            "deno:/opt/deno",
        )

    command = run.call_args.args[0]
    assert command[3:11] == [
        "--js-runtimes", "deno:/opt/deno",
        "--extractor-args", "youtube:player_client=android",
        "--cookies", str(cookies),
        "--ffmpeg-location", str(Path(sys.executable).parent),
    ]
    assert command[11:14] == ["--no-playlist", "--no-progress", "--no-overwrites"]
    assert result["path"] == "youtube__x.mp4"
