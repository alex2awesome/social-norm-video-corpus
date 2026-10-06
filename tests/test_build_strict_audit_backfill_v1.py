import json
import sqlite3
from pathlib import Path

import pytest

from scripts.build_strict_audit_backfill_v1 import build


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def touch_video(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"video")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_build_separates_media_and_labels_and_joins_old_coverage(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    instr = root / "data/instructional/u1"
    touch_video(instr / "demo_0.mp4")
    write_json(
        instr / "metadata.json",
        {
            "video_id": "u1",
            "demos": [{"clip": "demo_0.mp4", "norm": "take turns", "polarity": "correct"}],
        },
    )
    hit = root / "data/hits/u2"
    touch_video(hit / "clip_3.mp4")
    write_json(
        hit / "metadata.json",
        {
            "video_id": "u2",
            "reactions": [
                {"clip_idx": 3, "norm": "do not cut", "phrase": "hey", "clip_window": [1, 2]},
                {"clip_idx": 3, "norm": "wait in line", "phrase": "stop", "clip_window": [1, 2]},
            ],
        },
    )
    touch_video(root / "data/discussion_video/u3.mp4")
    write_json(
        root / "data/discussion/u3.json",
        {
            "video_id": "u3",
            "statements": [
                {"quote": "That was rude", "norm": "be polite", "start": 2, "end": 3},
                {"quote": "He interrupted", "norm": "do not interrupt", "start": 4, "end": 5},
            ],
        },
    )
    touch_video(root / "data/negatives/u4/negative_0.mp4")

    old = root / "data/shadow_scores/20260724_full_corpus_v1"
    old.mkdir(parents=True)
    manifest = {
        "item_id": "instructional:u1:0",
        "source_clip": str(instr / "demo_0.mp4"),
    }
    (old / "instructional_manifest.jsonl").write_text(json.dumps(manifest) + "\n")
    score = {"item_id": "instructional:u1:0", "error": None, "low_level": {"motion_mean": 1}}
    (old / "instructional_low_level_canonical.jsonl").write_text(json.dumps(score) + "\n")
    for pillar in ("witnessed", "commentary"):
        (old / f"{pillar}_manifest.jsonl").write_text("")
        (old / f"{pillar}_low_level_canonical.jsonl").write_text("")

    out = tmp_path / "run"
    summary = build(root, out, root / "missing.db")
    assert summary["media_items"] == 4
    assert summary["weak_label_items"] == 5
    assert summary["media_by_pillar"] == {
        "commentary": 1,
        "instructional": 1,
        "negative": 1,
        "witnessed": 1,
    }
    media = {row["pillar"]: row for row in rows(out / "media_manifest.jsonl")}
    assert media["instructional"]["low_level_visual_complete"] is True
    assert media["witnessed"]["label_count"] == 2
    assert media["commentary"]["label_count"] == 2
    assert len(rows(out / "strict_audit_queue.jsonl")) == 5


def test_manual_audit_requires_all_labels_for_media(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    touch_video(root / "data/discussion_video/u.mp4")
    write_json(
        root / "data/discussion/u.json",
        {
            "video_id": "u",
            "statements": [
                {"quote": "q1", "norm": "n1"},
                {"quote": "q2", "norm": "n2"},
            ],
        },
    )
    db = root / "data/visual_audit/audit.db"
    db.parent.mkdir(parents=True)
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE commentary_judgments(item_id TEXT, decision TEXT)")
    connection.execute("INSERT INTO commentary_judgments VALUES('commentary:u:0','accept')")
    connection.commit()
    connection.close()

    out = tmp_path / "run"
    summary = build(root, out, db)
    assert summary["labels_with_manual_strict_audit"]["commentary"] == 1
    assert summary["media_with_all_labels_manually_audited"]["commentary"] == 0
    assert summary["media_needing_strict_audit"]["commentary"] == 1


def test_refuses_to_overwrite_frozen_run(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    out = tmp_path / "run"
    out.mkdir()
    with pytest.raises(FileExistsError):
        build(root, out, root / "missing.db")


def test_includes_append_only_low_level_shards_and_calibration_reviews(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    instr = root / "data/instructional/u1"
    touch_video(instr / "demo_0.mp4")
    write_json(
        instr / "metadata.json",
        {
            "video_id": "u1",
            "demos": [{"clip": "demo_0.mp4", "norm": "take turns", "polarity": "correct"}],
        },
    )
    shard = root / "data/shadow_scores/20260811_strict_audit_backfill_queues_v1/low_level_shard_0_of_2.jsonl"
    shard.parent.mkdir(parents=True)
    shard.write_text(
        json.dumps(
            {
                "item_id": "instructional:u1:0",
                "source_clip": str(instr / "demo_0.mp4"),
                "low_level": {"motion_mean": 0.1},
                "error": None,
            }
        )
        + "\n"
    )
    audit = root / "audit_runs/20260811_strict_audit_calibration_v2"
    audit.mkdir(parents=True)
    (audit / "sealed_selection.jsonl").write_text(
        json.dumps({"audit_index": 0, "item_id": "instructional:u1:0"}) + "\n"
    )
    (audit / "manual_post_reveal_review.jsonl").write_text(
        json.dumps({"audit_index": 0, "strict_decision": "accept"}) + "\n"
    )
    out = tmp_path / "run"
    summary = build(root, out, root / "missing.db")
    assert summary["labels_with_low_level_visual"]["instructional"] == 1
    assert summary["labels_with_manual_strict_audit"]["instructional"] == 1
    assert summary["strict_audit_queue_items"] == 0
