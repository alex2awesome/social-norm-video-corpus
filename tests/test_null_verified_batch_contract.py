from pathlib import Path

from src import search_loop


def _run_null_route(monkeypatch, saved):
    calls = {}
    monkeypatch.setattr(
        search_loop.clip_extract,
        "save_null_verified",
        lambda *args, **kwargs: saved,
        raising=False,
    )
    monkeypatch.setattr(
        search_loop.state,
        "finalize_null",
        lambda conn, uid, count: calls.update(finalized=(conn, uid, count)),
    )
    monkeypatch.setattr(
        search_loop.scrape,
        "purge_raw",
        lambda uid, cfg: calls.update(purged=(uid, cfg)),
    )
    cfg = {"loop": {"purge_raw_on_miss": True}}
    result = search_loop.finish_detection(
        "db", cfg, "video-1", {"url": "https://example.test/video-1"},
        Path("video.mp4"), "ordinary routine", "null_home", "null", {},
        [], "drop:no_violation", "candid", "detector", [], None,
    )
    return result, calls


def test_null_verified_saved_windows_are_counted_before_sqlite(monkeypatch):
    saved = [
        {"neg_idx": 0, "window": [0.0, 22.0]},
        {"neg_idx": 1, "window": [22.0, 44.0]},
    ]
    result, calls = _run_null_route(monkeypatch, saved)
    assert result == 2
    assert calls["finalized"] == ("db", "video-1", 2)
    assert calls["purged"][0] == "video-1"


def test_null_verified_empty_result_stores_zero_without_purging(monkeypatch):
    result, calls = _run_null_route(monkeypatch, [])
    assert result == 0
    assert calls["finalized"] == ("db", "video-1", 0)
    assert "purged" not in calls


def test_saved_window_contract_rejects_a_scalar_count():
    try:
        search_loop.saved_window_count(2)
    except TypeError as exc:
        assert "must return a list" in str(exc)
    else:
        raise AssertionError("scalar extractor result should not silently pass")


def test_batch_wrapper_stops_when_detection_stage_fails():
    script = Path("batch_pipeline.sh").read_text()
    assert 'if ! "$AI_PY" -u -m src.batch_detect; then' in script
    assert "ERROR stage 2 (detect) failed; stopping pipeline" in script
