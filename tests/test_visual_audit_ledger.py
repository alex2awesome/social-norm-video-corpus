import json
import sqlite3

import pytest

from scripts.visual_audit_ledger import (
    connect,
    import_instructional_repair_judgments,
    is_duplicate_integrity_error,
    trim_label_transcript,
    transform_has_audio_removal,
    validate_instructional_repair_judgment,
)


def repair_row():
    return {
        "item_id": "instructional:test:0",
        "batch_id": "repair_batch",
        "rubric_version": "instructional_repair_v1",
        "model": "manual-test",
        "prompt_sha256": "prompt",
        "source_frame_manifest_sha256": "source",
        "repaired_frame_manifest_sha256": "repaired",
        "repair_type": "tighten_to_demo",
        "transform": {"start_sec": 2.0, "end_sec": 8.0, "source_duration_sec": 10.0},
        "is_social_norm": "yes",
        "visual_demo_present": "yes",
        "norm_supported": "yes",
        "polarity_supported": "violation",
        "label_leak_visible": "no",
        "narration_leak": "no",
        "decision": "accept",
        "required_repairs": [],
        "normalized_behavior": "One person insults another person.",
        "normalized_norm": "People should not insult others.",
        "rejection_reasons": [],
        "evidence_frames": [0, 1],
        "description": "Rendered output contains the complete exchange.",
    }


def test_repair_acceptance_requires_valid_rendered_bounds():
    row = repair_row()
    validate_instructional_repair_judgment(row)
    row["transform"]["end_sec"] = 11.0
    with pytest.raises(ValueError, match="invalid tighten_to_demo bounds"):
        validate_instructional_repair_judgment(row)


def test_foreign_key_failure_is_not_counted_as_duplicate(tmp_path):
    results = tmp_path / "results.jsonl"
    results.write_text(json.dumps(repair_row()) + "\n")
    conn = connect(tmp_path / "audit.db")
    with pytest.raises(ValueError, match="FOREIGN KEY constraint failed"):
        import_instructional_repair_judgments(conn, results)
    conn.close()


def test_only_unique_collisions_are_resumable_skips():
    duplicate = sqlite3.IntegrityError("UNIQUE constraint failed: judgments.item_id")
    foreign_key = sqlite3.IntegrityError("FOREIGN KEY constraint failed")
    assert is_duplicate_integrity_error(duplicate)
    assert not is_duplicate_integrity_error(foreign_key)


def test_crop_repair_requires_recorded_geometry():
    row = repair_row()
    row["repair_type"] = "crop_label_overlay"
    row["transform"] = {
        "crop_geometry": "1920:900:0:0",
        "audio_removed": True,
        "source_audio_stream_count": 1,
        "repaired_audio_stream_count": 0,
    }
    validate_instructional_repair_judgment(row)
    row["transform"]["crop_geometry"] = ""
    with pytest.raises(ValueError, match="crop_label_overlay requires crop_geometry"):
        validate_instructional_repair_judgment(row)


def test_recut_preserves_nested_audio_removal_provenance():
    transform = {
        "prior_transform": {
            "prior_transform": {"audio_removed": True},
            "output_audio_removed": False,
        }
    }
    assert transform_has_audio_removal(transform)
    assert not transform_has_audio_removal({"prior_transform": {}})


def test_recut_preserves_transcript_across_multiple_repairs():
    first = {
        "aligned_transcript": [],
        "source_label_transcript": [
            {"clip_start": 3.0, "clip_end": 12.0, "text": "comparison"}
        ],
    }
    once = trim_label_transcript(first, 2.0, 10.0)
    assert once == [
        {
            "source_clip_start": 3.0,
            "source_clip_end": 12.0,
            "recut_start": 1.0,
            "recut_end": 10.0,
            "text": "comparison",
        }
    ]
    twice = trim_label_transcript(
        {"aligned_transcript": [], "source_label_transcript": once}, 0.5, 7.5
    )
    assert twice[0]["recut_start"] == 0.5
    assert twice[0]["recut_end"] == 9.5
    assert twice[0]["text"] == "comparison"


def test_repair_acceptance_allows_polarity_only_relabel_with_supported_norm():
    row = repair_row()
    row["decision"] = "accept_after_relabel"
    row["required_repairs"] = ["relabel_polarity"]
    row["norm_supported"] = "yes"
    validate_instructional_repair_judgment(row)

    row["norm_supported"] = "no"
    with pytest.raises(ValueError, match="polarity-only relabel requires norm_supported=yes"):
        validate_instructional_repair_judgment(row)


def test_repair_norm_relabel_requires_unsupported_original_norm():
    row = repair_row()
    row["decision"] = "accept_after_relabel"
    row["required_repairs"] = ["relabel_norm"]
    row["norm_supported"] = "no"
    validate_instructional_repair_judgment(row)

    row["norm_supported"] = "yes"
    with pytest.raises(ValueError, match="norm relabel requires norm_supported=no"):
        validate_instructional_repair_judgment(row)
