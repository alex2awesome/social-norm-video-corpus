import json
import sqlite3
from pathlib import Path

from scripts.export_corpus_shadow_manifest import (
    export_rows,
    source_disjoint_stratified_sample,
)
from scripts.render_corpus_shadow_proxies import (
    latest_by_item,
    sampling_fps,
    successful_by_item,
    write_complete_manifest,
)
from scripts.score_social_norm_text_gate import successful_item_ids


def test_export_rows_is_stable_and_uses_existing_clips(tmp_path: Path):
    clip = tmp_path / "data" / "instructional" / "u1" / "demo.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"video")
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE items (
        item_id TEXT, pillar TEXT, uid TEXT, item_index INTEGER,
        metadata_path TEXT, clip_path TEXT, title TEXT, category TEXT,
        genre TEXT, source_platform TEXT, found_by_query TEXT,
        query_source TEXT, polarity TEXT, norm TEXT, start_quote TEXT,
        end_quote TEXT, explanation TEXT, start_sec REAL, end_sec REAL,
        source_mtime REAL, present INTEGER, has_clip INTEGER)"""
    )
    conn.execute(
        """INSERT INTO items VALUES
        ('instructional:u1:0','instructional','u1',0,'metadata.json',
         'data/instructional/u1/demo.mp4','title','category',NULL,
         'dailymotion','query','seed','correct','courtesy','a','b','e',
         10,16,1,1,1)"""
    )
    rows = export_rows(conn, tmp_path, "instructional")
    assert [row["item_id"] for row in rows] == ["instructional:u1:0"]
    assert rows[0]["duration_hint"] == 6
    assert rows[0]["source_exists"] is True
    assert rows[0]["ordinal"] == 0
    assert rows[0]["media_start_sec"] is None
    assert rows[0]["media_end_sec"] is None


def test_export_commentary_uses_retained_video_and_bounded_context(
    tmp_path: Path,
):
    clip = tmp_path / "data" / "discussion_video" / "u1.mp4"
    clip.parent.mkdir(parents=True)
    clip.write_bytes(b"video")
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """CREATE TABLE items (
        item_id TEXT, pillar TEXT, uid TEXT, item_index INTEGER,
        metadata_path TEXT, clip_path TEXT, title TEXT, category TEXT,
        genre TEXT, source_platform TEXT, found_by_query TEXT,
        query_source TEXT, polarity TEXT, norm TEXT, start_quote TEXT,
        end_quote TEXT, explanation TEXT, start_sec REAL, end_sec REAL,
        source_mtime REAL, present INTEGER, has_clip INTEGER)"""
    )
    conn.execute(
        """INSERT INTO items VALUES
        ('commentary:u1:0','commentary','u1',0,'metadata.json',
         NULL,'title','category','narration','dailymotion','query',
         'commentary','criticism','courtesy','a','b',NULL,
         10,14,1,1,0)"""
    )
    rows = export_rows(
        conn,
        tmp_path,
        "commentary",
        commentary_context_before=6,
        commentary_context_after=8,
    )
    assert len(rows) == 1
    assert rows[0]["source_clip"] == str(clip)
    assert rows[0]["source_exists"] is True
    assert rows[0]["media_start_sec"] == 4
    assert rows[0]["media_end_sec"] == 22
    assert rows[0]["duration_hint"] == 18


def test_failed_append_only_attempts_are_retried():
    records = [
        {"item_id": "a", "result": None, "error": "timeout"},
        {"item_id": "b", "result": {"ok": True}, "error": None},
    ]
    assert successful_item_ids(records) == {"b"}
    assert successful_by_item(
        [
            {"item_id": "a", "proxy_clip": None, "error": "bad"},
            {"item_id": "b", "proxy_clip": "/tmp/b.mp4", "error": None},
        ]
    ).keys() == {"b"}
    assert latest_by_item(
        [
            {"item_id": "a", "error": "first"},
            {"item_id": "a", "error": "second"},
        ]
    )["a"]["error"] == "second"


def test_complete_proxy_manifest_is_exact_and_non_overwriting(tmp_path: Path):
    source = [{"item_id": "a", "ordinal": 0, "pillar": "instructional"}]
    success = {
        "a": {
            "item_id": "a",
            "ordinal": 0,
            "proxy_clip": "/tmp/a.mp4",
            "error": None,
        }
    }
    target = tmp_path / "complete.jsonl"
    write_complete_manifest(source, success, target)
    first = json.loads(target.read_text())
    assert first["proxy_clip"] == "/tmp/a.mp4"
    write_complete_manifest(source, success, target)
    changed = {"a": {**success["a"], "proxy_clip": "/tmp/other.mp4"}}
    try:
        write_complete_manifest(source, changed, target)
    except ValueError:
        pass
    else:
        raise AssertionError("changed frozen manifest should not be overwritten")


def test_proxy_sampling_caps_frame_count():
    assert sampling_fps(10, 3, 96) == 3
    assert sampling_fps(100, 3, 96) == 0.96


def test_stratified_sample_is_deterministic_and_source_disjoint():
    rows = [
        {
            "item_id": f"i:{index}",
            "uid": f"u{index // 2}",
            "polarity": "correct" if index % 2 else "violation",
            "category": f"c{index % 3}",
        }
        for index in range(20)
    ]
    first = source_disjoint_stratified_sample([dict(x) for x in rows], 8, "seed")
    second = source_disjoint_stratified_sample([dict(x) for x in rows], 8, "seed")
    assert [row["item_id"] for row in first] == [row["item_id"] for row in second]
    assert len({row["uid"] for row in first}) == len(first) == 8
    assert [row["ordinal"] for row in first] == list(range(8))
