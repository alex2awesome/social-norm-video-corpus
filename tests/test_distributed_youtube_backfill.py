from __future__ import annotations

import json

from scripts.build_distributed_youtube_manifest import (
    WORKER_BUCKETS,
    assigned_worker,
    build_records,
    uid_bucket,
)
from scripts.distributed_youtube_backfill import is_auth_failure, load_records


def test_bucket_assignment_is_total_disjoint_and_deterministic():
    buckets = [bucket for values in WORKER_BUCKETS.values() for bucket in values]
    assert sorted(buckets) == list(range(20))
    assert assigned_worker("youtube__abc") == assigned_worker("youtube__abc")
    assert uid_bucket("youtube__abc") in range(20)


def test_build_records_excludes_present_and_assigns_worker(tmp_path):
    discussion = tmp_path / "discussion"
    discussion.mkdir()
    for uid in ("youtube__a", "youtube__b"):
        (discussion / f"{uid}.json").write_text(json.dumps({
            "video_id": uid, "url": f"https://youtube.test/{uid}",
        }))
    records = build_records(discussion, {"youtube__a"})
    assert [row["uid"] for row in records] == ["youtube__b"]
    assert records[0]["worker"] in WORKER_BUCKETS


def test_load_records_filters_worker(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join([
        json.dumps({"uid": "a", "worker": "sk1"}),
        json.dumps({"uid": "b", "worker": "sk2"}),
    ]))
    assert [row["uid"] for row in load_records(manifest, "sk2")] == ["b"]


def test_auth_failure_detection_is_narrow():
    assert is_auth_failure("Sign in to confirm you’re not a bot")
    assert is_auth_failure("ERROR LOGIN_REQUIRED")
    assert not is_auth_failure("Video unavailable")
