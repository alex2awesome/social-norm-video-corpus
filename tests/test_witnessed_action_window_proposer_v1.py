import json

import pytest

from scripts.witnessed_action_window_proposer_v1 import (
    joint_high_items,
    propose_action_window,
    review_sample,
    select_items,
)


def propose(**overrides):
    kwargs = dict(
        item_id="witnessed:u:clip_0",
        reactions=[{"start": 70.0, "clip_idx": 0}],
        clip_window=(58.0, 74.0),
        audio_events=None,
    )
    kwargs.update(overrides)
    return propose_action_window(**kwargs)


def test_basic_pre_reaction_window():
    row = propose()
    # anchor 70.0: end = 69.7, start = max(58, 60) = 60.
    assert row["status"] == "proposed"
    assert row["action_window_sec"] == [60.0, 69.7]
    assert row["action_window_clip_relative_sec"] == [2.0, 11.7]
    assert row["audio_snap"]["used"] is False
    assert row["media_cut"] is False and row["acceptance_label"] is None


def test_earliest_reaction_anchors_and_guard_excludes_reaction():
    row = propose(reactions=[{"start": 71.5}, {"start": 66.0}, {"start": None}])
    assert row["reaction_anchor_sec"] == 66.0
    assert row["action_window_sec"][1] == pytest.approx(65.7)
    assert row["n_reaction_timestamps"] == 2


def test_impact_peak_snaps_window_start():
    events = [
        {"class": "Crowd", "group": "commotion", "t0": 65, "t1": 66, "peak": 0.9},
        {"class": "Slap, smack", "group": "commotion", "t0": 64, "t1": 65, "peak": 0.41},
        {"class": "Smash, crash", "group": "commotion", "t0": 66, "t1": 67, "peak": 0.35},
        {"class": "Skidding", "group": "commotion", "t0": 72, "t1": 73, "peak": 0.9},
    ]
    row = propose(audio_events=events)
    # Latest qualifying impact BEFORE the 70.0 anchor is Smash at 66 -> start 64.
    assert row["audio_snap"]["used"] is True
    assert row["audio_snap"]["class"] == "Smash, crash"
    assert row["action_window_sec"][0] == 64.0
    # Ambient Crowd and post-anchor Skidding never snap.


def test_weak_or_ambient_impacts_do_not_snap():
    events = [
        {"class": "Slap, smack", "group": "commotion", "t0": 65, "t1": 66, "peak": 0.1},
        {"class": "Siren", "group": "commotion", "t0": 66, "t1": 67, "peak": 0.9},
    ]
    row = propose(audio_events=events)
    assert row["audio_snap"]["used"] is False
    assert row["action_window_sec"][0] == 60.0


def test_snap_never_shrinks_below_minimum_duration():
    events = [
        {"class": "Slap, smack", "group": "commotion", "t0": 69.5, "t1": 70, "peak": 0.9},
    ]
    row = propose(audio_events=events)
    # Snapped start would be 67.5 -> duration 2.2 >= 2.0: allowed.
    assert row["action_window_sec"][0] == 67.5
    events[0]["t0"] = 69.9  # snapped start 67.9 -> duration 1.8 < 2.0: fall back
    row = propose(audio_events=events)
    assert row["action_window_sec"][0] == 60.0
    assert row["audio_snap"]["used"] is False


def test_short_window_and_missing_timestamp_flags():
    row = propose(reactions=[{"start": 59.0}])
    assert row["status"] == "short_window_review"
    row = propose(reactions=[{"clip_idx": 0}])
    assert row["status"] == "no_reaction_timestamp"
    assert row["action_window_sec"] is None


def test_joint_high_items_and_review_sample(tmp_path):
    model = tmp_path
    def write(name, rows):
        (model / name).write_text("\n".join(json.dumps(r) for r in rows))
    write("posteriors_witnessed_norm_event_supported.jsonl", [
        {"item_id": "witnessed:a:clip_0", "shadow_band": "high_confidence_candidate", "posterior_positive": 0.99},
        {"item_id": "witnessed:b:clip_0", "shadow_band": "high_confidence_candidate", "posterior_positive": 0.97},
        {"item_id": "witnessed:c:clip_0", "shadow_band": "insufficient_evidence_abstain", "posterior_positive": 0.5},
    ])
    write("posteriors_witnessed_reaction_grounded.jsonl", [
        {"item_id": "witnessed:a:clip_0", "shadow_band": "high_confidence_candidate", "posterior_positive": 0.98},
        {"item_id": "witnessed:c:clip_0", "shadow_band": "high_confidence_candidate", "posterior_positive": 0.96},
    ])
    items = joint_high_items(model)
    assert set(items) == {"witnessed:a:clip_0"}
    assert items["witnessed:a:clip_0"]["norm_event"] == 0.99
    assert items["witnessed:a:clip_0"]["evidence_tier"] == "joint_high"

    everything = select_items(model, "any_high")
    assert set(everything) == {
        "witnessed:a:clip_0", "witnessed:b:clip_0", "witnessed:c:clip_0",
    }
    assert everything["witnessed:b:clip_0"]["evidence_tier"] == "norm_event_only"
    assert everything["witnessed:b:clip_0"]["reaction"] is None
    assert everything["witnessed:c:clip_0"]["evidence_tier"] == "reaction_only"
    with pytest.raises(ValueError, match="selection"):
        select_items(model, "everything")

    proposals = [
        propose(item_id=f"witnessed:u{i}:clip_0") for i in range(4)
    ] + [
        propose(item_id=f"witnessed:s{i}:clip_0",
                audio_events=[{"class": "Slap, smack", "t0": 64, "t1": 65, "peak": 0.9}])
        for i in range(4)
    ] + [
        propose(item_id="witnessed:short:clip_0", reactions=[{"start": 59.0}])
    ]
    sample = review_sample(proposals, 9, "seed")
    strata = {row["review_stratum"] for row in sample}
    assert strata == {"snapped", "unsnapped", "short"}
    assert all(row["review_complete"] is False for row in sample)
    # Deterministic.
    again = review_sample(proposals, 9, "seed")
    assert [r["item_id"] for r in sample] == [r["item_id"] for r in again]
