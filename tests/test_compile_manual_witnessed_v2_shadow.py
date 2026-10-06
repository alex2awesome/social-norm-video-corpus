from __future__ import annotations

from pathlib import Path

import pytest

from scripts.compile_manual_witnessed_v2_shadow import compile_rows


def row(item_id="witnessed:u:0"):
    return {
        "item_id": item_id,
        "actor_kind": "person",
        "behavior_kind": "speech_act",
        "affected_context_kind": "person",
        "expectation_kind": "interpersonal_treatment",
        "action_observed": "yes",
        "reaction_source_role": "bystander",
        "reaction_grounding": "visible_on_scene",
        "reaction_content": "targeted_objection",
        "temporal_relation": "action_established_before_reaction",
        "reaction_candidate_origin": "nearby_transcript_recovery",
        "current_label_relation": "repairable",
        "pre_reaction_demo_quality": "clear_visual",
        "boundary_basis": "exact_video",
        "authenticity": "organic",
        "action_end_sec": 3.0,
        "reaction_start_sec": 3.5,
        "action_evidence_frames": [0],
        "reaction_evidence_frames": [1],
        "description": "Recovered nearby objection.",
    }


def manifest():
    return {
        "batch_id": "b",
        "items": [
            {
                "item_id": "witnessed:u:0",
                "uid": "u",
                "duration": 8.0,
                "frame_manifest_sha256": "f",
                "frames": [{"frame_index": 0}, {"frame_index": 1}],
            }
        ],
    }


def test_compiler_derives_recovery_and_complete_coverage(tmp_path):
    source = tmp_path / "manual.jsonl"
    source.write_text("{}\n")
    records, summary = compile_rows(manifest(), [row()], source)
    assert records[0]["disposition"] == "recover_strict"
    assert summary["complete_manual_coverage"] is True
    assert summary["dispositions"] == {"recover_strict": 1}


def test_compiler_rejects_partial_manual_review(tmp_path):
    with pytest.raises(ValueError, match="cover manifest exactly"):
        compile_rows(manifest(), [], tmp_path / "manual.jsonl")
