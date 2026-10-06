import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "export_frozen_instructional_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location("frozen_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_motion_resolution_must_exactly_cover_ambiguous_rows():
    frames = [
        {"audit_index": 0, "uid": "a", "blind_scene_presence": "yes"},
        {"audit_index": 1, "uid": "b", "blind_scene_presence": "needs_motion"},
    ]
    with pytest.raises(ValueError, match="exactly cover"):
        MODULE.final_blind_decisions(frames, [])
    assert MODULE.final_blind_decisions(
        frames,
        [{"audit_index": 1, "uid": "b", "resolved_scene_presence": "no"}],
    ) == {(0, "a"): True, (1, "b"): False}


def test_new_blind_and_motion_field_names_are_supported():
    frames = [
        {"audit_index": 0, "uid": "a", "blind_scene": "yes"},
        {"audit_index": 1, "uid": "b", "blind_scene": "needs_motion"},
    ]
    assert MODULE.final_blind_decisions(
        frames,
        [{"audit_index": 1, "uid": "b", "resolved_blind_scene": "no"}],
    ) == {(0, "a"): True, (1, "b"): False}


def test_export_keeps_scene_label_and_relabel_targets_separate(tmp_path):
    manifest = {
        "visual_samples": [
            {
                "audit_index": 0,
                "uid": "a",
                "clip_index": 2,
                "clip_path": "/remote/a.mp4",
                "norm": "be kind",
                "title": "A",
                "category": "c",
                "polarity": "correct",
                "explanation": "e",
                "start_quote": "s",
                "end_quote": "t",
            },
            {
                "audit_index": 1,
                "uid": "b",
                "clip_index": 0,
                "clip_path": "/remote/b.mp4",
                "norm": "vague value",
                "title": "B",
                "category": "c",
                "polarity": "violation",
                "explanation": "e",
                "start_quote": "s",
                "end_quote": "t",
            },
        ]
    }
    frames = [
        {"audit_index": 0, "uid": "a", "blind_scene_presence": "yes"},
        {"audit_index": 1, "uid": "b", "blind_scene_presence": "no"},
    ]
    labels = [
        {
            "audit_index": 0,
            "uid": "a",
            "proposed_norm": "be kind",
            "disposition": "accept_exact",
            "source_recut_candidate": False,
        },
        {
            "audit_index": 1,
            "uid": "b",
            "proposed_norm": "vague value",
            "disposition": "accept_relabel",
            "relabel": "return the wallet",
            "source_recut_candidate": False,
        },
    ]
    rows = MODULE.export_rows(manifest, frames, [], labels, tmp_path)
    assert rows[0]["gold_scene_visible"] is True
    assert rows[0]["gold_label_matched_visible"] is True
    assert rows[0]["gold_usable"] is True
    assert rows[0]["is_social_norm"] == "yes"
    assert rows[1]["gold_scene_visible"] is False
    assert rows[1]["gold_label_matched_visible"] is False
    assert rows[1]["gold_usable"] is True
    assert rows[1]["gold_relabel"] == "return the wallet"
    assert rows[1]["is_social_norm"] == "yes"


def test_export_supports_strict_new_adjudication_schema(tmp_path):
    manifest = {
        "visual_samples": [
            {
                "audit_index": 0,
                "uid": "a",
                "clip_index": 2,
                "clip_path": "/remote/a.mp4",
                "norm": "vague value",
                "polarity": "explanation",
            },
            {
                "audit_index": 1,
                "uid": "b",
                "clip_index": 0,
                "clip_path": "/remote/b.mp4",
                "norm": "technical task",
                "polarity": "correct",
            },
        ]
    }
    frames = [
        {"audit_index": 0, "uid": "a", "blind_scene": "yes"},
        {"audit_index": 1, "uid": "b", "blind_scene": "no"},
    ]
    labels = [
        {
            "audit_index": 0,
            "uid": "a",
            "final_scene": "yes",
            "adjudication": "accept_relabel",
            "relabel_norm": "do not insult a customer",
            "relabel_polarity": "violation",
            "source_recut_candidate": False,
        },
        {
            "audit_index": 1,
            "uid": "b",
            "final_scene": "no",
            "adjudication": "reject_nonsocial",
            "source_recut_candidate": False,
        },
    ]
    rows = MODULE.export_rows(manifest, frames, [], labels, tmp_path)
    assert rows[0]["gold_usable"] is True
    assert rows[0]["gold_relabel"] == "do not insult a customer"
    assert rows[0]["gold_relabel_polarity"] == "violation"
    assert rows[1]["gold_usable"] is False
    assert rows[1]["is_social_norm"] == "no"


def test_new_schema_cannot_accept_a_visually_absent_scene(tmp_path):
    manifest = {
        "visual_samples": [
            {
                "audit_index": 0,
                "uid": "a",
                "clip_index": 0,
                "clip_path": "/remote/a.mp4",
                "norm": "be kind",
            }
        ]
    }
    frames = [{"audit_index": 0, "uid": "a", "blind_scene": "no"}]
    labels = [
        {
            "audit_index": 0,
            "uid": "a",
            "final_scene": "no",
            "adjudication": "accept_exact",
            "source_recut_candidate": False,
        }
    ]
    with pytest.raises(ValueError, match="cannot accept"):
        MODULE.export_rows(manifest, frames, [], labels, tmp_path)
