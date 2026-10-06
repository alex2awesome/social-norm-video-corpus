import json

from scripts.labeling_functions_v1 import validate_lf_record
from scripts.visual_feature_lfs_v1 import (
    commentary_visual_lf_records,
    instructional_visual_lf_records,
    load_feature_index,
    witnessed_visual_lf_records,
)


def clip_scores(staged, real, talking, news, slide, broll):
    return {"mean_probabilities": [staged, real, talking, news, slide, broll]}


def by_lf(records):
    for record in records:
        validate_lf_record(record)
    return {r["lf_id"]: r for r in records}


def test_witnessed_crowd_scene_supports_bystander_possibility():
    features = {
        "keypoints": {"person_count_mean": 3.6, "person_present_fraction": 0.92},
        "clip_scores": clip_scores(0.05, 0.62, 0.10, 0.05, 0.02, 0.08),
    }
    lfs = by_lf(witnessed_visual_lf_records("witnessed:u:clip_0", features))
    assert lfs["vis_witnessed_person_count_v1"]["vote"] == 1
    assert lfs["vis_witnessed_person_count_v1"]["target"] == "independent_bystander_signal"
    assert lfs["vis_witnessed_interaction_scene_v1"]["vote"] == 1
    assert lfs["vis_witnessed_staged_roleplay_v1"]["vote"] == 0


def test_witnessed_solo_talking_head_votes_negative():
    features = {
        "keypoints": {"person_count_mean": 1.0, "person_present_fraction": 0.9},
        "clip_scores": clip_scores(0.03, 0.05, 0.81, 0.40, 0.02, 0.05),
    }
    lfs = by_lf(witnessed_visual_lf_records("witnessed:u:clip_0", features))
    # A solo scene cannot contain an independent bystander.
    assert lfs["vis_witnessed_person_count_v1"]["vote"] == -1
    assert lfs["vis_witnessed_interaction_scene_v1"]["vote"] == -1


def test_witnessed_staged_roleplay_excludes_organic_subtype():
    features = {
        "keypoints": {},
        "clip_scores": clip_scores(0.71, 0.30, 0.05, 0.02, 0.01, 0.03),
    }
    lfs = by_lf(witnessed_visual_lf_records("witnessed:u:clip_1", features))
    assert lfs["vis_witnessed_staged_roleplay_v1"]["vote"] == -1
    assert lfs["vis_witnessed_person_count_v1"]["vote"] == 0


def test_instructional_roleplay_is_positive_demo_evidence():
    # Staged role-play is a DEMO for instructional (any performed format).
    features = {
        "keypoints": {"multiple_people_fraction": 0.7, "person_present_fraction": 0.95},
        "clip_scores": clip_scores(0.66, 0.20, 0.15, 0.05, 0.02, 0.10),
    }
    lfs = by_lf(instructional_visual_lf_records("instructional:u:demo_0", features))
    assert lfs["vis_instructional_demo_scene_v1"]["vote"] == 1
    assert lfs["vis_instructional_person_pair_v1"]["vote"] == 1


def test_instructional_slides_and_broll_vote_negative():
    features = {
        "keypoints": {"multiple_people_fraction": 0.0, "person_present_fraction": 0.1},
        "clip_scores": clip_scores(0.02, 0.05, 0.10, 0.15, 0.83, 0.20),
    }
    lfs = by_lf(instructional_visual_lf_records("instructional:u:demo_1", features))
    assert lfs["vis_instructional_demo_scene_v1"]["vote"] == -1
    assert lfs["vis_instructional_person_pair_v1"]["vote"] == -1
    missing = by_lf(instructional_visual_lf_records("instructional:u:demo_2", {"keypoints": {}}))
    assert missing["vis_instructional_demo_scene_v1"]["vote"] == 0


def test_commentary_low_level_only_and_static_graphic_negative():
    active = {"low_level": {"face_present_fraction": 0.8, "multiple_faces_fraction": 0.4,
                            "motion_mean": 0.2}}
    static = {"low_level": {"face_present_fraction": 0.0, "multiple_faces_fraction": 0.0,
                            "motion_mean": 0.01}}
    (row,) = commentary_visual_lf_records("commentary:u:stmt_0", active)
    validate_lf_record(row)
    assert row["vote"] == 1 and row["target"] == "event_present_in_source"
    (row,) = commentary_visual_lf_records("commentary:u:stmt_0", static)
    assert row["vote"] == -1


def test_load_feature_index_parses_media_filename(tmp_path):
    path = tmp_path / "features.jsonl"
    rows = [
        {"uid": "u1", "pillar": "witnessed", "clip": "/x/data/hits/u1/clip_2.mp4",
         "low_level": {"motion_mean": 0.1}, "keypoints": {"person_count_mean": 2.0},
         "clip_scores": None, "error": None},
        {"uid": "u2", "pillar": "witnessed", "clip": "/x/u2/clip_0.mp4",
         "low_level": {}, "error": "decode failed"},
        {"uid": "c1", "pillar": "commentary", "clip": "/x/discussion_video/c1.mp4",
         "low_level": {"face_present_fraction": 0.5}, "error": None},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    index = load_feature_index(path)
    assert ("witnessed", "u1", 2) in index
    assert ("witnessed", "u2", 0) not in index  # errored rows are dropped
    assert ("commentary", "c1", 0) in index
    assert load_feature_index(tmp_path / "missing.jsonl") == {}
