import pytest

from scripts.export_commentary_unlabeled_benchmark import composed_video_filter
from scripts.materialize_commentary_caption_crop_experiment import (
    CROP_VARIANTS,
    derive_trials,
)


def artifacts():
    sealed = [{
        "audit_index": 4,
        "candidate_id": "parent-4",
        "source_event_bounds_sec": [10.0, 16.0],
        "source_path": "/data/a.mp4",
        "uid": "a",
        "title": "hidden",
    }]
    blind = [{
        "candidate_id": "parent-4",
        "performed_event_visible": "yes",
        "actor_target_grounded": "yes",
        "label_bearing_text_absent": "no",
    }]
    post = [{
        "candidate_id": "parent-4",
        "exact_named_action_visible": "yes",
        "label_alignment": "exact",
        "route": "commentary_visual",
    }]
    return sealed, blind, post


def test_eligible_parent_receives_every_fixed_variant():
    trials = derive_trials(*artifacts())
    assert len(trials) == len(CROP_VARIANTS)
    assert {row["variant"] for row in trials} == set(CROP_VARIANTS)
    assert {row["parent_candidate_id"] for row in trials} == {"parent-4"}
    assert [row["audit_index"] for row in trials] == list(range(len(CROP_VARIANTS)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("performed_event_visible", "uncertain"),
        ("actor_target_grounded", "no"),
        ("label_bearing_text_absent", "yes"),
    ],
)
def test_visual_contract_fails_closed(field, value):
    sealed, blind, post = artifacts()
    blind[0][field] = value
    with pytest.raises(ValueError, match="no exact caption-leaking"):
        derive_trials(sealed, blind, post)


@pytest.mark.parametrize(
    "field,value",
    [
        ("exact_named_action_visible", "uncertain"),
        ("label_alignment", "same_norm"),
        ("route", "relabel_required"),
    ],
)
def test_semantic_contract_fails_closed(field, value):
    sealed, blind, post = artifacts()
    post[0][field] = value
    with pytest.raises(ValueError, match="no exact caption-leaking"):
        derive_trials(sealed, blind, post)


def test_mismatched_artifact_populations_are_rejected():
    sealed, blind, post = artifacts()
    post[0]["candidate_id"] = "different"
    with pytest.raises(ValueError, match="do not cover the same"):
        derive_trials(sealed, blind, post)


def test_composed_filter_places_crop_between_sampling_and_scale():
    crop = CROP_VARIANTS["bands_08_20"]
    assert composed_video_filter(crop) == f"fps=3,{crop},scale='min(640,iw)':-2"


@pytest.mark.parametrize("bad", ["", "crop=1:1;movie=x", "[in]crop=1:1[out]"])
def test_composed_filter_rejects_unsafe_or_graph_filters(bad):
    with pytest.raises(ValueError, match="invalid spatial filter"):
        composed_video_filter(bad)
