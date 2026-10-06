from pathlib import Path

import pytest

from scripts.render_commentary_hierarchical_windows_v2 import (
    create_source_links,
    manual_templates,
    prepare_specs,
)


def selected(window_id="w", uid="dailymotion__a", ordinal=0):
    return {
        "window_id": window_id,
        "item_id": window_id,
        "source_item_id": f"commentary:{uid}:0",
        "uid": uid,
        "source_path": "/data/source.mp4",
        "source_duration_sec": 30,
        "title": "Video shows customer hitting worker",
        "action_label": "customer hits worker",
        "window_ordinal": ordinal,
        "window_start_sec": 8,
        "window_end_sec": 20,
        "selection_reasons": ["feature_motion"],
        "cheap_features": {"motion": 0.5},
        "cheap_feature_error": None,
    }


def test_prepare_specs_assigns_stable_audit_indices_and_preserves_lineage():
    later = selected("later", "youtube__z", 3)
    earlier = selected("earlier", "dailymotion__a", 1)
    specs, semantic, lineage = prepare_specs([later, earlier])
    assert semantic[0]["candidate_id"] == "earlier"
    assert specs[0] == {"audit_index": 0, "start_sec": 8.0, "end_sec": 20.0}
    assert lineage["later"]["source_item_id"] == "commentary:youtube__z:0"
    assert lineage["earlier"]["selection_reasons"] == ["feature_motion"]


def test_prepare_specs_rejects_scorer_lineage_mismatch_and_bad_bounds():
    row = selected()
    row["item_id"] = "wrong"
    with pytest.raises(ValueError, match="scorer item lineage"):
        prepare_specs([row])
    row = selected()
    row["window_end_sec"] = 31
    with pytest.raises(ValueError, match="bounds"):
        prepare_specs([row])


def test_source_links_do_not_copy_media_and_must_not_change_target(tmp_path: Path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    row = selected()
    row["source_path"] = str(source)
    _, _, lineage = prepare_specs([row])
    links = tmp_path / "links"
    create_source_links(lineage, links)
    link = links / "w.mp4"
    assert link.is_symlink()
    assert link.resolve() == source
    source_two = tmp_path / "other.mp4"
    source_two.write_bytes(b"other")
    lineage["w"]["source_path"] = str(source_two)
    with pytest.raises(ValueError, match="points elsewhere"):
        create_source_links(lineage, links)


def test_manual_templates_exactly_cover_every_selected_window():
    _, _, lineage = prepare_specs([
        selected("a", "dailymotion__a", 0),
        selected("b", "youtube__b", 1),
    ])
    blind, post = manual_templates(lineage)
    assert {row["window_id"] for row in blind} == {"a", "b"}
    assert {row["window_id"] for row in post} == {"a", "b"}
    assert all(row["performed_event_visible"] == "" for row in blind)
    assert all(row["exact_named_action_visible"] == "" for row in post)
