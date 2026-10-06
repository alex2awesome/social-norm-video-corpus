import time
from pathlib import Path

import pytest

from src import state


def config(tmp_path: Path):
    return {
        "paths": {"state_db": str(tmp_path / "state.db")},
        "scheduler": {
            "platform_weights": {"youtube": 1.0, "dailymotion": 1.0},
            "family_weights": {
                "instructional": 1.0,
                "witnessed": 1.0,
                "commentary": 1.0,
                "negative": 0.1,
            },
            "blocked_query_patterns": [r"\bpolice\b", r"\bcops?\b"],
            "zero_new_cooldown": {
                "enabled": True,
                "base_seconds": 600,
                "pagination_base_seconds": 60,
                "source_failure_seconds": 30,
                "max_seconds": 2400,
            },
        },
    }


def test_infers_families_and_reversibly_excludes_blocked_theme(tmp_path: Path):
    cfg = config(tmp_path)
    conn = state.init_db(cfg)
    assert state.infer_query_family("instructional", "instr_family") == "instructional"
    assert state.infer_query_family("commentary", None) == "commentary"
    assert state.infer_query_family("taxonomy", "wit_queue") == "witnessed"
    assert state.infer_query_family("null", "null_queue") == "negative"

    assert not state.add_query(
        conn, "youtube", "police confrontation", "typical_norm_v1", cfg=cfg
    )
    conn.execute(
        "INSERT INTO queries(platform,query,source,active,added_at) VALUES(?,?,?,?,?)",
        ("youtube", "cop argues with driver", "legacy", 1, time.time()),
    )
    state._backfill_query_families(conn)
    state._apply_query_policy(conn, cfg)
    row = conn.execute(
        "SELECT active,policy_excluded,policy_reason FROM queries WHERE query=?",
        ("cop argues with driver",),
    ).fetchone()
    assert row["active"] == 1  # reversible policy exclusion, not deactivation
    assert row["policy_excluded"] == 1
    assert row["policy_reason"].startswith("blocked_theme:")


def test_family_balancing_happens_before_platform_and_priority(tmp_path: Path):
    cfg = config(tmp_path)
    conn = state.init_db(cfg)
    for family, category in (
        ("instructional", "instr_home"),
        ("witnessed", "wit_queue"),
        ("commentary", "comm_service"),
    ):
        assert state.add_query(
            conn, "youtube", f"{family} ordinary query", "typical_norm_v1",
            priority=3.25, category=category, family=family, cfg=cfg,
        )
    now = time.time()
    conn.execute("UPDATE queries SET last_used=? WHERE family='instructional'", (now - 300,))
    conn.execute("UPDATE queries SET last_used=? WHERE family='witnessed'", (now - 200,))
    conn.execute("UPDATE queries SET last_used=? WHERE family='commentary'", (now - 100,))
    conn.commit()
    assert state.next_query(conn, cfg)["family"] == "instructional"


def test_zero_new_cooldown_is_exponential_and_success_resets_it(tmp_path: Path):
    cfg = config(tmp_path)
    conn = state.init_db(cfg)
    state.add_query(
        conn, "youtube", "roommate chore role play", "typical_norm_v1",
        category="instr_home", family="instructional", cfg=cfg,
    )
    clock = time.time()
    first = state.record_query_run(
        conn, "youtube", "roommate chore role play", started_at=clock - 2,
        candidates_returned=50, new_candidates=0, enqueued_candidates=0,
        skipped_candidates=0, cursor_before=None, cursor_after=None,
        cfg=cfg, now=clock,
    )
    assert first["zero_new_streak"] == 1
    assert first["cooldown_until"] == pytest.approx(clock + 600)
    assert state.next_query(conn, cfg) is None
    assert state.query_cooldown_wait(conn, cfg, now=clock) == pytest.approx(600)

    second_clock = clock + 601
    second = state.record_query_run(
        conn, "youtube", "roommate chore role play", started_at=second_clock - 2,
        candidates_returned=50, new_candidates=0, enqueued_candidates=0,
        skipped_candidates=0, cursor_before=None, cursor_after=None,
        cfg=cfg, now=second_clock,
    )
    assert second["zero_new_streak"] == 2
    assert second["cooldown_until"] == pytest.approx(second_clock + 1200)

    success_clock = second_clock + 1201
    success = state.record_query_run(
        conn, "youtube", "roommate chore role play", started_at=success_clock - 2,
        candidates_returned=12, new_candidates=3, enqueued_candidates=2,
        skipped_candidates=1, cursor_before=None, cursor_after=None,
        cfg=cfg, now=success_clock,
    )
    assert success["zero_new_streak"] == 0
    assert success["cooldown_until"] is None
    row = conn.execute(
        "SELECT zero_new_streak,cooldown_until,last_new_at FROM queries"
    ).fetchone()
    assert row["zero_new_streak"] == 0
    assert row["cooldown_until"] is None
    assert row["last_new_at"] == pytest.approx(success_clock)
    assert conn.execute("SELECT count(*) FROM query_runs").fetchone()[0] == 3


def test_source_failure_throttles_without_claiming_query_saturation(tmp_path: Path):
    cfg = config(tmp_path)
    conn = state.init_db(cfg)
    state.add_query(
        conn, "dailymotion", "holding door etiquette", "typical_norm_v1",
        category="instr_public", family="instructional", cfg=cfg,
    )
    clock = time.time()
    result = state.record_query_run(
        conn, "dailymotion", "holding door etiquette", started_at=clock - 1,
        candidates_returned=0, new_candidates=0, enqueued_candidates=0,
        skipped_candidates=0, cursor_before="2", cursor_after=None,
        run_status="source_unavailable", cfg=cfg, now=clock,
    )
    assert result["zero_new_streak"] == 0
    assert result["cooldown_until"] == pytest.approx(clock + 30)


def test_skip_reason_is_separate_from_error_text(tmp_path: Path):
    conn = state.init_db(config(tmp_path))
    candidate = {
        "uid": "youtube__one",
        "source": "youtube",
        "title": "ordinary etiquette demo",
        "channel": "example",
        "duration": 60,
        "url": "https://example.test/one",
    }
    assert state.enumerate_video(
        conn, candidate, "door etiquette", "youtube", "instr_public", "instructional"
    )
    state.set_status(
        conn, candidate["uid"], "skipped", skip_reason="download_failed_filtered_or_unavailable"
    )
    row = conn.execute(
        "SELECT status,error,skip_reason FROM seen_videos WHERE video_id=?",
        (candidate["uid"],),
    ).fetchone()
    assert dict(row) == {
        "status": "skipped",
        "error": None,
        "skip_reason": "download_failed_filtered_or_unavailable",
    }
