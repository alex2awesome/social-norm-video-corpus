import json
import sqlite3
from pathlib import Path

from scripts.visual_audit_sample import (
    commentary_rows,
    exclude_prior_sources,
    exclude_uids,
    extract_frames,
    load_uid_exclusions,
    prior_audit_source_uids,
)


def make_ledger(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE items(item_id TEXT PRIMARY KEY, pillar TEXT, uid TEXT);
        CREATE TABLE batch_items(batch_id TEXT, item_id TEXT);
        CREATE TABLE judgments(item_id TEXT);
        CREATE TABLE commentary_judgments(item_id TEXT);
        """
    )
    conn.execute(
        "INSERT INTO items VALUES (?,?,?)",
        ("instructional:video-a:0", "instructional", "video-a"),
    )
    conn.execute(
        "INSERT INTO items VALUES (?,?,?)",
        ("instructional:video-a:1", "instructional", "video-a"),
    )
    conn.execute(
        "INSERT INTO items VALUES (?,?,?)",
        ("commentary:video-b:0", "commentary", "video-b"),
    )
    conn.execute(
        "INSERT INTO batch_items VALUES (?,?)",
        ("batch-1", "instructional:video-a:0"),
    )
    conn.execute(
        "INSERT INTO commentary_judgments VALUES (?)",
        ("commentary:video-b:0",),
    )
    conn.commit()
    conn.close()


def test_prior_assignment_or_judgment_excludes_whole_source_video(tmp_path):
    db = tmp_path / "audit.db"
    make_ledger(db)
    prior = prior_audit_source_uids(db)
    assert prior == {("instructional", "video-a"), ("commentary", "video-b")}
    rows = [
        {"pillar": "instructional", "uid": "video-a", "clip_index": 1},
        {"pillar": "instructional", "uid": "video-c", "clip_index": 0},
    ]
    kept, excluded = exclude_prior_sources(rows, prior)
    assert kept == [{"pillar": "instructional", "uid": "video-c", "clip_index": 0}]
    assert excluded == 1


def test_missing_ledger_is_fail_open_for_sampling_only(tmp_path):
    assert prior_audit_source_uids(tmp_path / "missing.db") == set()


def test_explicit_uid_exclusions_are_union_and_source_wide(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("# prior review\nvideo-a\n")
    second.write_text("video-b\nvideo-a\n")
    excluded = load_uid_exclusions([first, second])
    rows = [
        {"uid": "video-a", "clip_index": 0},
        {"uid": "video-a", "clip_index": 1},
        {"uid": "video-c", "clip_index": 0},
    ]
    kept, count = exclude_uids(rows, excluded)
    assert excluded == {"video-a", "video-b"}
    assert kept == [{"uid": "video-c", "clip_index": 0}]
    assert count == 2


def test_commentary_sampler_requires_and_joins_retained_video(tmp_path):
    discussion = tmp_path / "discussion"
    videos = tmp_path / "discussion_video"
    discussion.mkdir()
    videos.mkdir()
    (discussion / "with-video.json").write_text(
        json.dumps(
            {
                "video_id": "with-video",
                "title": "A visible source",
                "statements": [{"norm": "queue cutting", "quote": "He cut in line"}],
            }
        )
    )
    (discussion / "text-only.json").write_text(
        json.dumps({"video_id": "text-only", "statements": []})
    )
    media = videos / "with-video.mp4"
    media.write_bytes(b"not-empty")

    rows = commentary_rows(discussion, cutoff=None, video_root=videos)

    assert len(rows) == 1
    assert rows[0]["uid"] == "with-video"
    assert rows[0]["clip_path"] == str(media)
    assert rows[0]["statement"] == "He cut in line"


def test_newly_retained_video_makes_old_commentary_record_fresh(tmp_path):
    discussion = tmp_path / "discussion"
    videos = tmp_path / "discussion_video"
    discussion.mkdir()
    videos.mkdir()
    metadata = discussion / "x.json"
    metadata.write_text(json.dumps({"video_id": "x"}))
    media = videos / "x.mp4"
    media.write_bytes(b"video")
    cutoff = (metadata.stat().st_mtime + media.stat().st_mtime) / 2

    assert commentary_rows(discussion, cutoff=cutoff, video_root=videos)


def test_old_commentary_json_is_not_parsed_before_recency_filter(
    monkeypatch, tmp_path
):
    discussion = tmp_path / "discussion"
    videos = tmp_path / "discussion_video"
    discussion.mkdir()
    videos.mkdir()
    old = discussion / "old.json"
    old.write_text("{not valid json")
    media = videos / "old.mp4"
    media.write_bytes(b"video")
    cutoff = max(old.stat().st_mtime, media.stat().st_mtime) + 1

    def fail(_path):
        raise AssertionError("old metadata should not be parsed")

    monkeypatch.setattr("scripts.visual_audit_sample.load_json", fail)
    assert commentary_rows(discussion, cutoff=cutoff, video_root=videos) == []


def test_temporal_frame_extraction_uses_accurate_output_side_seek(
    monkeypatch, tmp_path
):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"video")
    observed = []

    monkeypatch.setattr(
        "scripts.visual_audit_sample.probe_duration",
        lambda _ffprobe, _clip: 10.0,
    )

    class Result:
        returncode = 0
        stderr = ""

    def fake_run(command, **_kwargs):
        observed.append(command)
        Path(command[-1]).write_bytes(b"jpg")
        return Result()

    monkeypatch.setattr("scripts.visual_audit_sample.subprocess.run", fake_run)
    row = {"uid": "x", "pillar": "instructional", "clip_path": str(clip)}
    frames = extract_frames(
        row, 0, tmp_path, [0.5], "ffmpeg", "ffprobe"
    )

    assert frames
    assert observed[0].index("-i") < observed[0].index("-ss")
