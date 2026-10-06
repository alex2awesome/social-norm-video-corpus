import math

import numpy as np

from scripts.witnessed_reaction_identity_features import (
    FaceObservation,
    acoustic_identity_features,
    box_iou,
    gaze_proxy,
    intervention_features,
    link_face_observations,
    role_separation_features,
    speaker_turn_features,
    summarize_face_tracks,
    temporal_structure_features,
)


def obs(time, x, value=100):
    crop = np.full((12, 12), value, dtype=np.uint8)
    return FaceObservation(time, (x, 0, 10, 10), crop, gaze_proxy(crop))


def test_intervention_subtypes_are_nonexclusive():
    result = intervention_features("Call security and ask if she is okay")
    assert result["intervention.delegate"] == 1
    assert result["intervention.support"] == 1
    assert result["intervention.any_active"] == 1


def test_generic_affect_is_not_active_intervention():
    result = intervention_features("Oh wow!")
    assert result["intervention.generic_affect_only"] == 1
    assert result["intervention.any_active"] == 0


def test_explicit_speaker_change_and_role_difference():
    segments = [
        {"start": 0, "end": 1, "text": "move", "speaker": "A"},
        {"start": 1.1, "end": 2, "text": "stop", "speaker": "B"},
    ]
    result = speaker_turn_features(segments, 1, action_speaker="A")
    assert result["speaker.explicit_available"] == 1
    assert result["speaker.changed_at_reaction"] == 1
    assert result["speaker.reactor_differs_from_action"] == 1


def test_speaker_features_abstain_without_diarization():
    result = speaker_turn_features([{"start": 0, "end": 1, "text": "stop"}], 0)
    assert result["speaker.explicit_available"] == 0
    assert math.isnan(result["speaker.changed_at_reaction"])


def test_temporal_structure_measures_latency():
    segments = [
        {"start": 0, "end": 1, "text": "action"},
        {"start": 1.4, "end": 2, "text": "stop that"},
    ]
    result = temporal_structure_features(segments, 1)
    assert abs(result["structure.response_latency"] - 0.4) < 1e-8
    assert result["structure.short_response"] == 1


def test_role_separation_distinguishes_target_and_third_party():
    third = role_separation_features("subject", "bystander")
    target = role_separation_features("subject", "victim")
    assert third["roles.reactor_third_party"] == 1
    assert target["roles.reactor_affected_target"] == 1


def test_box_iou_and_face_linking():
    assert box_iou((0, 0, 10, 10), (1, 0, 10, 10)) > 0.8
    tracks = link_face_observations([[obs(0, 0)], [obs(1, 1)]])
    assert len(tracks) == 1
    assert len(tracks[0].observations) == 2


def test_face_summary_detects_cross_boundary_change_and_synchrony():
    frames = [
        [obs(0.0, 0, 100), obs(0.0, 30, 100)],
        [obs(0.5, 1, 100), obs(0.5, 31, 100)],
        [obs(1.0, 15, 180), obs(1.0, 45, 180)],
        [obs(1.5, 16, 180), obs(1.5, 46, 180)],
    ]
    result = summarize_face_tracks(link_face_observations(frames, minimum_iou=0), 1.0)
    assert result["face_tracks.cross_boundary_tracks"] == 2
    assert result["face_tracks.synchronous_reactors"] == 2


def test_acoustic_features_abstain_without_media(tmp_path):
    result = acoustic_identity_features(tmp_path / "none.mp4", 2.0)
    assert result["acoustic.proxy_available"] == 0
