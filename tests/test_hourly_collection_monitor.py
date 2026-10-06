from __future__ import annotations

import json
import sqlite3

from scripts.hourly_collection_monitor import (
    audited_search_status,
    backfill_gap_status,
    count_lines,
    fresh_review_queue,
    ledger_summary,
    shadow_score_status,
    score_item_status,
    source_inventory,
    state_status_counts,
    strict_audit_backfill_status,
    successful_item_ids,
    witnessed_video_asr_render_status,
)


def test_inventory_counts_unique_media_and_missing(tmp_path):
    discussion = tmp_path / "discussion"
    videos = tmp_path / "discussion_video"
    discussion.mkdir()
    videos.mkdir()
    for uid in ("dailymotion__a", "dailymotion__b"):
        (discussion / f"{uid}.json").write_text("{}")
    (videos / "dailymotion__a.mp4").write_bytes(b"video")
    (videos / "dailymotion__a.info.json").write_text("{}")
    result = source_inventory(discussion, videos)
    assert result["dailymotion"] == {"records": 2, "retained": 1, "missing": 1}


def test_ledger_uses_latest_status_per_uid(tmp_path):
    ledger = tmp_path / "backfill.jsonl"
    rows = [
        {"uid": "dailymotion__a", "status": "failed", "auth_failure": False},
        {"uid": "dailymotion__a", "status": "downloaded"},
        {"uid": "dailymotion__b", "status": "failed", "auth_failure": True},
    ]
    ledger.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    result = ledger_summary(ledger)
    assert result["unique_downloaded"] == 1
    assert result["latest_failed"] == 1
    assert result["auth_failures_total"] == 1


def test_backfill_gap_status_separates_terminal_failure_from_unattempted(tmp_path):
    discussion = tmp_path / "discussion"
    videos = tmp_path / "discussion_video"
    discussion.mkdir()
    videos.mkdir()
    for uid in ("dailymotion__a", "dailymotion__b", "dailymotion__c"):
        (discussion / f"{uid}.json").write_text("{}")
    (videos / "dailymotion__a.mp4").write_bytes(b"video")
    ledger = videos / "backfill.jsonl"
    ledger.write_text(
        json.dumps({"uid": "dailymotion__b", "status": "failed"}) + "\n"
    )
    result = backfill_gap_status(discussion, videos, ledger)
    assert result["dailymotion"] == {
        "missing": 2,
        "latest_failed": 1,
        "unattempted": 1,
        "other_latest_status": 0,
    }


def test_review_queue_does_not_repeat_previously_queued_uid(tmp_path):
    discussion = tmp_path / "discussion"
    videos = tmp_path / "discussion_video"
    out = tmp_path / "monitor"
    discussion.mkdir()
    videos.mkdir()
    out.mkdir()
    for uid in ("dailymotion__a", "dailymotion__b"):
        (discussion / f"{uid}.json").write_text(json.dumps({"title": uid}))
        (videos / f"{uid}.mp4").write_bytes(uid.encode())
    (out / "snapshots.jsonl").write_text(
        json.dumps({"fresh_review_queue": [{"uid": "dailymotion__a"}]}) + "\n"
    )
    result = fresh_review_queue(discussion, videos, out / "snapshots.jsonl", 4)
    assert [row["uid"] for row in result] == ["dailymotion__b"]
    assert result[0]["label_status"] == "source_only_unlabeled"


def test_state_status_counts_reads_database_without_writing(tmp_path):
    import sqlite3

    database = tmp_path / "state.db"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE seen_videos (status TEXT)")
    connection.executemany(
        "INSERT INTO seen_videos VALUES (?)",
        [("pending_detect",), ("pending_detect",), ("done",)],
    )
    connection.commit()
    connection.close()

    assert state_status_counts(database) == {
        "done": 1,
        "pending_detect": 2,
    }
    assert not (tmp_path / "state.db-wal").exists()


def test_audited_search_status_reports_queries_and_outputs(tmp_path):
    database = tmp_path / "state.db"
    connection = sqlite3.connect(database)
    connection.execute(
        """CREATE TABLE queries (
            query TEXT, category TEXT, priority REAL, active INTEGER,
            source TEXT, videos_processed INTEGER, videos_with_hits INTEGER,
            total_reaction_count INTEGER, last_used REAL
        )"""
    )
    connection.execute(
        """CREATE TABLE seen_videos (
            video_id TEXT, title TEXT, query TEXT, category TEXT, status TEXT,
            modality TEXT, query_source TEXT, enumerated_at REAL,
            processed_at REAL
        )"""
    )
    connection.execute(
        "INSERT INTO queries VALUES (?,?,?,?,?,?,?,?,?)",
        ("acted scene", "instr_scene", 3.25, 1, "manual_audit_search_v1", 2, 1, 1, 4),
    )
    connection.execute(
        "INSERT INTO seen_videos VALUES (?,?,?,?,?,?,?,?,?)",
        (
            "dailymotion__a",
            "Acted Scene",
            "acted scene",
            "instr_scene",
            "done",
            "instructional",
            "manual_audit_search_v1",
            3,
            5,
        ),
    )
    connection.commit()
    connection.close()
    result = audited_search_status(database)
    assert result["active_queries"] == 1
    assert result["videos_processed"] == 2
    assert result["videos_with_hits"] == 1
    assert result["outcomes"] == [
        {"status": "done", "modality": "instructional", "items": 1}
    ]
    assert result["recent_candidates"][0]["uid"] == "dailymotion__a"


def test_shadow_score_status_counts_progress_and_processes(tmp_path):
    run = tmp_path / "shadow_scores" / "20260724_full_corpus_v1"
    run.mkdir(parents=True)
    for pillar, total, attempted in (
        ("instructional", 3, 2),
        ("witnessed", 2, 2),
        ("commentary", 4, 0),
    ):
        (run / f"{pillar}_manifest.jsonl").write_text("{}\n" * total)
        if attempted:
            (run / f"{pillar}_low_level.jsonl").write_text(
                "".join(
                    json.dumps(
                        {
                            "item_id": f"{pillar}:{index}",
                            "low_level": {"motion_mean": 0},
                            "error": None,
                        }
                    )
                    + "\n"
                    for index in range(attempted)
                )
            )

    patterns = []

    def fake_process_count(pattern):
        patterns.append(pattern)
        return int("instructional" in pattern)

    result = shadow_score_status(tmp_path, fake_process_count)
    assert result is not None
    assert result["run"] == "20260724_full_corpus_v1"
    assert result["pillars"]["instructional"]["remaining_items"] == 1
    assert result["pillars"]["instructional"]["processes"] == 1
    assert result["pillars"]["witnessed"]["complete"] is True
    assert result["pillars"]["commentary"]["progress_fraction"] == 0
    assert len(patterns) == 3


def test_shadow_score_status_deduplicates_base_and_shards(tmp_path):
    run = tmp_path / "shadow_scores" / "20260724_full_corpus_v1"
    run.mkdir(parents=True)
    (run / "instructional_manifest.jsonl").write_text("{}\n" * 3)
    for pillar in ("witnessed", "commentary"):
        (run / f"{pillar}_manifest.jsonl").write_text("")
    record_a = {
        "item_id": "instructional:a",
        "low_level": {"motion_mean": 0},
        "error": None,
    }
    record_b = {
        "item_id": "instructional:b",
        "low_level": {"motion_mean": 1},
        "error": None,
    }
    (run / "instructional_low_level.jsonl").write_text(json.dumps(record_a) + "\n")
    (run / "instructional_low_level_shard_00.jsonl").write_text(
        json.dumps(record_a) + "\n" + json.dumps(record_b) + "\n"
    )
    result = shadow_score_status(tmp_path, lambda _: 2)
    assert result["pillars"]["instructional"]["attempted_items"] == 2
    assert result["pillars"]["instructional"]["remaining_items"] == 1
    assert len(result["pillars"]["instructional"]["outputs"]) == 2


def test_successful_item_ids_ignores_errors_and_empty_features(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        '{"item_id":"a","low_level":{"x":1},"error":null}\n'
        '{"item_id":"b","low_level":{},"error":null}\n'
        '{"item_id":"c","low_level":{"x":1},"error":"bad"}\n'
        'not-json\n'
    )
    assert successful_item_ids([path]) == {"a"}
    assert score_item_status([path]) == {"a": True, "b": False, "c": False}


def test_score_item_status_successful_retry_wins(tmp_path):
    failed = tmp_path / "failed.jsonl"
    retried = tmp_path / "retried.jsonl"
    failed.write_text('{"item_id":"a","low_level":{},"error":"decode"}\n')
    retried.write_text(
        '{"item_id":"a","low_level":{"motion_mean":1},"error":null}\n'
    )
    assert score_item_status([failed, retried]) == {"a": True}


def test_shadow_score_status_tracks_pose_clip_backfill(tmp_path):
    run = tmp_path / "shadow_scores" / "20260724_full_corpus_v1"
    multimodal = run / "multimodal_v1"
    multimodal.mkdir(parents=True)
    for pillar in ("instructional", "witnessed", "commentary"):
        (run / f"{pillar}_manifest.jsonl").write_text(
            "{}\n" * (2 if pillar != "commentary" else 0)
        )
    (multimodal / "instructional_pose_clip.jsonl").write_text(
        json.dumps(
            {
                "item_id": "instructional:a",
                "low_level": {"motion": 1},
                "keypoints": {"people": 2},
                "clip_scores": {"probabilities": [0.8]},
                "error": None,
            }
        )
        + "\n"
    )
    (multimodal / "witnessed_pose_clip.jsonl").write_text(
        json.dumps(
            {
                "item_id": "witnessed:a",
                "low_level": {"motion": 1},
                "keypoints": None,
                "clip_scores": {"probabilities": [0.8]},
                "error": None,
            }
        )
        + "\n"
    )
    result = shadow_score_status(tmp_path, lambda pattern: int("instructional" in pattern))
    assert result["multimodal_v1"]["instructional"]["attempted_items"] == 1
    assert result["multimodal_v1"]["instructional"]["successfully_scored_items"] == 1
    assert result["multimodal_v1"]["instructional"]["remaining_items"] == 1
    assert result["multimodal_v1"]["instructional"]["processes"] == 1
    assert result["multimodal_v1"]["witnessed"]["failed_items"] == 1
    assert result["multimodal_v1"]["witnessed"]["processes"] == 0


def test_commentary_progress_excludes_superseded_random_seek_output(tmp_path):
    run = tmp_path / "shadow_scores" / "20260724_full_corpus_v1"
    run.mkdir(parents=True)
    for pillar in ("instructional", "witnessed"):
        (run / f"{pillar}_manifest.jsonl").write_text("")
    (run / "commentary_manifest.jsonl").write_text("{}\n" * 2)
    template = {"low_level": {"motion_mean": 1}, "error": None}
    (run / "commentary_low_level.jsonl").write_text(
        json.dumps({"item_id": "old", **template}) + "\n"
    )
    (run / "commentary_low_level_v2_sequential.jsonl").write_text(
        json.dumps({"item_id": "new", **template}) + "\n"
    )
    result = shadow_score_status(tmp_path, lambda _: 0)
    status = result["pillars"]["commentary"]
    assert status["attempted_items"] == 1
    assert "commentary_low_level.jsonl" not in status["outputs"]


def test_count_lines_handles_missing_and_unterminated_last_record(tmp_path):
    missing = tmp_path / "missing.jsonl"
    assert count_lines(missing) == 0
    path = tmp_path / "records.jsonl"
    path.write_bytes(b"{}\n{}\n{}")
    assert count_lines(path) == 2


def test_strict_audit_backfill_status_tracks_sharded_progress(tmp_path):
    coverage = tmp_path / "shadow_scores" / "coverage"
    queues = tmp_path / "shadow_scores" / "queues"
    coverage.mkdir(parents=True)
    queues.mkdir(parents=True)
    (coverage / "summary.json").write_text(
        json.dumps(
            {
                "media_by_pillar": {
                    "instructional": 5,
                    "witnessed": 3,
                    "commentary": 2,
                    "negative": 7,
                }
            }
        )
    )
    (queues / "summary.json").write_text(
        json.dumps({"low_level_delta_items": 3, "strict_vlm_items": 9})
    )
    good = {"item_id": "a", "low_level": {"motion": 1}, "error": None}
    bad = {"item_id": "b", "low_level": {}, "error": "decode"}
    (queues / "low_level_shard_0_of_2.jsonl").write_text(
        json.dumps(good) + "\n" + json.dumps(bad) + "\n"
    )
    result = strict_audit_backfill_status(
        tmp_path,
        lambda _: 2,
        coverage_run="coverage",
        queue_run="queues",
    )
    assert result["positive_media_items"] == 10
    assert result["strict_vlm_queue_items"] == 9
    assert result["low_level_attempted_items"] == 2
    assert result["low_level_successful_items"] == 1
    assert result["low_level_remaining_items"] == 1
    assert result["low_level_processes"] == 2
    assert result["low_level_stalled"] is False


def test_witnessed_video_asr_render_status_tracks_proxy_phase(tmp_path):
    run = (
        tmp_path
        / "shadow_scores"
        / "20260806_witnessed_video_asr_v3"
    )
    proxies = run / "full_proxies"
    media = run / "media"
    proxies.mkdir(parents=True)
    media.mkdir()
    (run / "job_spec.json").write_text(
        json.dumps(
            {"expected_source_clips": 3, "expected_candidate_windows": 5}
        )
    )
    (proxies / "a.mp4").write_bytes(b"proxy")
    (proxies / "zero.mp4").write_bytes(b"")
    (proxies / "ignore.txt").write_text("not media")

    patterns = []

    def fake_process_count(pattern):
        patterns.append(pattern)
        return 1

    result = witnessed_video_asr_render_status(tmp_path, fake_process_count)
    assert result is not None
    assert result["phase"] == "full_clip_proxy_render"
    assert result["full_clip_proxies"]["files"] == 1
    assert result["proxy_progress_fraction"] == 1 / 3
    assert result["candidate_progress_fraction"] == 0
    assert result["stalled"] is False
    assert result["automatic_acceptance"] is False
    assert len(patterns) == 1


def test_witnessed_video_asr_render_status_detects_completion_and_failures(tmp_path):
    run = (
        tmp_path
        / "shadow_scores"
        / "20260806_witnessed_video_asr_v3"
    )
    (run / "full_proxies").mkdir(parents=True)
    media = run / "media"
    media.mkdir()
    (run / "job_spec.json").write_text(
        json.dumps(
            {"expected_source_clips": 1, "expected_candidate_windows": 2}
        )
    )
    (run / "full_proxies" / "a.mp4").write_bytes(b"proxy")
    for name in ("a.mp4", "b.mp4"):
        (media / name).write_bytes(b"candidate")
    (run / "manifest.jsonl").write_text("{}\n{}\n")
    (run / "failures.jsonl").write_text("{}\n")
    (run / "render_summary.json").write_text(
        json.dumps({"selected_for_shard": 2, "failed": 1})
    )

    result = witnessed_video_asr_render_status(tmp_path, lambda _: 0)
    assert result is not None
    assert result["phase"] == "complete_with_failures_or_missing"
    assert result["complete"] is True
    assert result["successfully_complete"] is False
    assert result["reported_failures"] == 1
    assert result["failure_rows"] == 1
    assert result["stalled"] is False


def test_witnessed_video_asr_render_status_is_stalled_without_process(tmp_path):
    run = (
        tmp_path
        / "shadow_scores"
        / "20260806_witnessed_video_asr_v3"
    )
    run.mkdir(parents=True)
    (run / "job_spec.json").write_text(
        json.dumps(
            {"expected_source_clips": 1, "expected_candidate_windows": 1}
        )
    )
    result = witnessed_video_asr_render_status(tmp_path, lambda _: 0)
    assert result is not None
    assert result["stalled"] is True


def test_witnessed_video_asr_render_status_prefers_candidate_phase_after_proxy_failures(tmp_path):
    run = tmp_path / "shadow_scores" / "20260806_witnessed_video_asr_v3"
    proxies = run / "full_proxies"
    media = run / "media"
    proxies.mkdir(parents=True)
    media.mkdir()
    (run / "job_spec.json").write_text(
        json.dumps({"expected_source_clips": 3, "expected_candidate_windows": 4})
    )
    for name in ("a.mp4", "b.mp4"):
        (proxies / name).write_bytes(b"proxy")
    (media / "candidate.mp4").write_bytes(b"window")
    result = witnessed_video_asr_render_status(tmp_path, lambda _: 1)
    assert result["proxy_progress_fraction"] == 2 / 3
    assert result["phase"] == "candidate_window_render"


def test_witnessed_video_asr_render_status_tracks_coverage_retry(tmp_path):
    run = tmp_path / "shadow_scores" / "20260806_witnessed_video_asr_v3"
    run.mkdir(parents=True)
    (run / "job_spec.json").write_text(
        json.dumps({"expected_source_clips": 2, "expected_candidate_windows": 3})
    )
    (run / "render_summary.json").write_text(
        json.dumps({"selected_for_shard": 2, "rendered": 2})
    )
    result = witnessed_video_asr_render_status(tmp_path, lambda _: 1)
    assert result["coverage_retry_in_progress"] is True
    assert result["complete"] is False
    assert result["phase"] == "coverage_retry"
