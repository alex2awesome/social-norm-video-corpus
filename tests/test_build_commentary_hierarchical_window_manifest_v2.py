import pytest

from scripts.build_commentary_hierarchical_window_manifest_v2 import (
    SCORE_FIELDS,
    build_windows,
    select_diverse,
    tiled_bounds,
    validate_full_coverage,
)


def source(uid="dailymotion__a", duration=30.0, anchors=None):
    return {
        "uid": uid,
        "item_id": f"commentary:{uid}:0",
        "source_path": f"/data/{uid}.mp4",
        "duration_sec": duration,
        "title": "Video shows a customer hitting a worker",
        "action_label": "customer hits worker",
        "anchors": anchors or [],
    }


def scores(windows, error_ids=()):
    output = []
    for index, row in enumerate(windows):
        output.append({
            "window_id": row["window_id"],
            "error": "decode" if row["window_id"] in error_ids else None,
            "motion": float(index),
            "scene_change": float(len(windows) - index),
            "person_interaction": float(index % 3),
            "social_scene_similarity": float(index % 4),
        })
    return output


def test_tiling_covers_short_source_and_tail_without_gaps():
    assert tiled_bounds(5, 12, 8) == [(0.0, 5.0)]
    bounds = tiled_bounds(30, 12, 8)
    assert bounds[0] == (0.0, 12.0)
    assert bounds[-1] == (18.0, 30.0)
    assert all(right[0] <= left[1] for left, right in zip(bounds, bounds[1:]))


def test_build_windows_preserves_full_source_lineage_and_anchor_kinds():
    rows = build_windows([source(anchors=[{
        "start_sec": 9,
        "end_sec": 10,
        "kind": "visual_deixis",
    }])])
    validate_full_coverage(rows)
    assert any("visual_deixis" in row["anchor_kinds"] for row in rows)
    assert all(row["item_id"] == row["window_id"] for row in rows)
    assert all(row["source_item_id"] == "commentary:dailymotion__a:0" for row in rows)
    assert all(row["media_start_sec"] == row["window_start_sec"] for row in rows)
    assert all(row["candidate_generation_only"] is True for row in rows)
    assert all(row["automatic_acceptance"] is False for row in rows)


def test_sources_must_be_source_disjoint_and_well_formed():
    with pytest.raises(ValueError, match="duplicate uid"):
        build_windows([source(), source()])
    with pytest.raises(ValueError, match="over-limit"):
        build_windows([source(duration=1801)])
    bad = source(anchors=[{"start_sec": 1, "end_sec": 2, "kind": "magic"}])
    with pytest.raises(ValueError, match="anchor kind"):
        build_windows([bad])


def test_scores_must_exactly_cover_every_generated_window():
    windows = build_windows([source()])
    with pytest.raises(ValueError, match="exactly cover"):
        select_diverse(windows, scores(windows)[:-1])
    malformed = scores(windows)
    del malformed[0][SCORE_FIELDS[0]]
    with pytest.raises(ValueError, match="invalid or missing"):
        select_diverse(windows, malformed)


def test_diverse_selection_is_bounded_and_keeps_multiple_mechanisms():
    windows = build_windows([source(duration=90, anchors=[
        {"start_sec": 8, "end_sec": 12, "kind": "prior_vlm"},
        {"start_sec": 40, "end_sec": 44, "kind": "commentary_statement"},
        {"start_sec": 70, "end_sec": 74, "kind": "visual_deixis"},
    ])])
    selected, summary = select_diverse(
        windows, scores(windows), per_route=2, max_per_source=10
    )
    assert 1 <= len(selected) <= 10
    reasons = {reason for row in selected for reason in row["selection_reasons"]}
    assert "anchor_visual_deixis" in reasons
    assert "anchor_commentary_statement" in reasons
    assert "anchor_prior_vlm" in reasons
    assert "feature_motion" in reasons
    assert "uniform_temporal_coverage" in reasons
    assert summary["all_selected_vlm_outputs_require_manual_audit"] is True
    assert summary["automatic_acceptance"] is False


def test_feature_errors_fail_closed_but_do_not_break_uniform_fallback():
    windows = build_windows([source(duration=20)])
    rows = scores(windows, {row["window_id"] for row in windows})
    selected, summary = select_diverse(windows, rows, per_route=1)
    assert selected
    assert summary["cheap_feature_score_coverage"] == 0.0
    assert all(row["cheap_feature_error"] == "decode" for row in selected)


def test_two_sources_remain_source_disjoint_in_output():
    windows = build_windows([
        source("dailymotion__a", 30),
        source("youtube__b", 40),
    ])
    selected, summary = select_diverse(windows, scores(windows), per_route=1)
    assert {row["uid"] for row in selected} == {"dailymotion__a", "youtube__b"}
    assert summary["sources"] == 2
