from pathlib import Path

import pytest

from scripts.materialize_commentary_clip_plan import (
    MIN_AUDIT_FRAMES,
    sample_rendered_frames,
    select_plans,
)


def row(index=0, uid="dailymotion__a"):
    return {
        "approval_status": "unreviewed_candidate",
        "audit_index": index,
        "candidate_id": f"opaque-{index}",
        "item_id": f"commentary:{uid}:0",
        "uid": uid,
        "source_path": f"/data/{uid}.mp4",
        "proposed_start_sec": 2,
        "proposed_end_sec": 8,
        "source_duration_sec": 10,
        "title": "hidden",
        "selection_rule": "dual_retrieval_exact_clear_validation",
        "proposal_source": "shorter_interval_plus_context",
    }


def test_select_plans_requires_unique_sources_and_sorts_by_audit_index():
    selected = select_plans([row(3, "dailymotion__b"), row(1, "youtube__a")])
    assert [value["audit_index"] for value in selected] == [1, 3]


def test_select_plans_rejects_duplicate_source():
    with pytest.raises(ValueError, match="duplicate uid"):
        select_plans([row(0), row(1)])


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("approval_status", "accepted", "must be unreviewed"),
        ("proposed_end_sec", 12, "invalid proposed bounds"),
        ("source_path", "/data/a.txt", "unsupported source extension"),
    ],
)
def test_select_plans_fails_closed_on_invalid_contract(field, value, match):
    value_row = row()
    value_row[field] = value
    with pytest.raises(ValueError, match=match):
        select_plans([value_row])


def test_short_but_auditable_clip_floor_is_eight_frames():
    assert MIN_AUDIT_FRAMES == 8


def test_renderer_uses_sequential_bounded_decode():
    observed = []

    def sampler(*args):
        observed.append(args)
        return [], {}

    sample_rendered_frames(
        Path("clip.mp4"),
        {"video_duration_sec": 6.0}, sampler=sampler,
    )
    assert observed == [(Path("clip.mp4"), 24, 0.0, 6.0)]
