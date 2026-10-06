from pathlib import Path

from scripts.score_commentary_title_event_cues import (
    build_candidates,
    title_cues,
)


def test_title_cues_require_concrete_event_relation():
    assert "capture_before_event" in title_cues(
        "Caught on camera: customer attacks worker"
    )
    assert "actor_action" in title_cues("Driver hits cyclist and flees")
    assert title_cues("Attorney discusses workplace bullying") == []
    assert title_cues("Amazing camera footage of the sun") == []


def test_build_candidates_requires_retained_media_and_excludes_animals(tmp_path: Path):
    media = tmp_path / "media"
    media.mkdir()
    (media / "good.mp4").write_bytes(b"x")
    (media / "good.info.json").write_text("{}")
    (media / "animal.mp4").write_bytes(b"x")
    rows = [
        {
            "uid": "good",
            "title": "Driver hits cyclist",
            "agent": "human",
            "source": "dailymotion",
            "query_source": "q",
            "found_by_query": "query",
        },
        {
            "uid": "animal",
            "title": "Dog attacks man caught on camera",
            "agent": "human",
            "source": "dailymotion",
            "query_source": "q",
            "found_by_query": "query",
        },
        {
            "uid": "missing",
            "title": "Driver hits cyclist",
            "agent": "human",
            "source": "dailymotion",
            "query_source": "q",
            "found_by_query": "query",
        },
    ]
    candidates = build_candidates(rows, media)
    assert [row["uid"] for row in candidates] == ["good"]
    assert candidates[0]["source_path"].endswith("/good.mp4")
