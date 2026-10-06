from scripts.score_transcript_visual_deixis import localize


def test_localize_requires_deixis_and_action_in_neighborhood() -> None:
    row = {
        "audit_index": 1,
        "candidate_id": "c1",
        "uid": "u1",
        "title": "title",
        "segments": [
            {"start": 10, "end": 12, "text": "Take a look at this video."},
            {"start": 12, "end": 15, "text": "He steals the package."},
        ],
    }
    result = localize(row)
    assert result["visual_deixis_window_count"] == 1
    assert result["visual_deixis_windows"][0]["start_sec"] == 8
    assert result["visual_deixis_windows"][0]["end_sec"] == 17
    assert "steals" in result["visual_deixis_windows"][0]["action_terms"]


def test_localize_does_not_treat_narrated_action_alone_as_visual_prior() -> None:
    row = {
        "audit_index": 1,
        "candidate_id": "c1",
        "uid": "u1",
        "title": "title",
        "segments": [
            {"start": 10, "end": 12, "text": "Police say he stole the package."}
        ],
    }
    assert localize(row)["visual_deixis_windows"] == []


def test_localize_does_not_treat_deixis_without_action_as_event() -> None:
    row = {
        "audit_index": 1,
        "candidate_id": "c1",
        "uid": "u1",
        "title": "title",
        "segments": [
            {"start": 10, "end": 12, "text": "Take a look at this person."}
        ],
    }
    assert localize(row)["visual_deixis_windows"] == []
