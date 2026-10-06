from scripts.witnessed_reaction_candidate_scan import (
    candidate_scan_features,
    scan_reaction_candidates,
    serialize_candidates,
)


def segments():
    return [
        {"clip_start": 1.0, "clip_end": 2.0, "text": "What are you doing?"},
        {"clip_start": 4.0, "clip_end": 5.0, "text": "Stop it."},
        {"clip_start": 7.0, "clip_end": 8.0, "text": "I was just joking."},
    ]


def test_scan_finds_active_candidates_across_entire_clip():
    candidates = scan_reaction_candidates(
        segments(), selected_quote="I was just joking."
    )
    assert [candidate.segment_index for candidate in candidates] == [0, 1]
    assert candidates[0].window_start == 0.0
    assert candidates[1].window_end == 8.0


def test_self_defense_is_retained_as_a_negative_flag_when_also_active():
    candidates = scan_reaction_candidates(
        [{"clip_start": 2, "clip_end": 3, "text": "Get away from me."}]
    )
    assert len(candidates) == 1
    assert candidates[0].negative_self_defense
    assert candidate_scan_features(candidates)["candidate_scan.active_nonnegative"] == 0


def test_generic_affect_abstains_unless_requested():
    rows = [{"start": 2, "end": 3, "text": "Wow!"}]
    assert scan_reaction_candidates(rows) == []
    candidates = scan_reaction_candidates(rows, include_generic_affect=True)
    assert len(candidates) == 1
    assert candidates[0].generic_affect_only


def test_candidate_features_count_location_and_mechanisms():
    candidates = scan_reaction_candidates(segments(), selected_quote="Stop it.")
    features = candidate_scan_features(candidates, selected_boundary=3.0)
    assert features["candidate_scan.total"] == 2
    assert features["candidate_scan.before_selected"] == 1
    assert features["candidate_scan.after_selected"] == 1
    assert features["candidate_scan.direct"] == 2
    assert features["candidate_scan.selected_quote_max_overlap"] == 1


def test_serialization_is_json_ready():
    row = serialize_candidates(scan_reaction_candidates(segments()))[0]
    assert row["segment_index"] == 0
    assert isinstance(row["mechanisms"], tuple)
