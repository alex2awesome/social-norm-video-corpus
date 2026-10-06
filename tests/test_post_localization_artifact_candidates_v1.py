import copy
import hashlib
import shutil
import subprocess

import pytest

from scripts.materialize_post_localization_candidates_v1 import (
    ffmpeg_argv,
    probe,
    ready_plans,
    validate_plan,
)
from scripts.validate_post_transform_artifact_review_v1 import validate


def plan():
    return {
        "candidate_id": "c", "pillar": "commentary", "uid": "u",
        "source_path": "/source.mp4", "proposed_start_sec": 1.0,
        "proposed_end_sec": 4.0, "behavior_label": "one person takes a bag",
        "ready_for_final_render": True,
        "source_sha256": "0" * 64, "source_lineage_sealed": True,
        "approval_status": "unreviewed_localization_candidate",
        "transform_required": "temporal_cut_and_mute",
    }


def test_renderer_plan_requires_ready_unapproved_muted_transform(tmp_path):
    row = plan()
    validate_plan([row])
    argv = ffmpeg_argv("ffmpeg", row, tmp_path / "out.mp4")
    assert "-an" in argv
    assert argv.index("-i") < argv.index("-ss")
    assert argv[argv.index("-t") + 1] == "3.000"
    bad = copy.deepcopy(row)
    bad["candidate_id"] = "d"
    bad["ready_for_final_render"] = False
    with pytest.raises(ValueError, match="not ready"):
        validate_plan([bad])
    assert ready_plans([row, bad]) == [row]


def test_instructional_render_requires_manual_audio_policy_review():
    row = plan()
    row["pillar"] = "instructional"
    with pytest.raises(ValueError, match="audio policy is unreviewed"):
        validate_plan([row])
    row["exact_boundary_and_audio_policy_review_complete"] = True
    validate_plan([row])


def review(value="yes"):
    return {
        "candidate_id": "c", "pillar": "commentary", "uid": "u",
        "visual_event_complete": value, "actor_target_grounded": "yes",
        "exact_behavior_label_supported": "yes", "start_boundary_clean": "yes",
        "end_boundary_clean": "yes", "label_bearing_visible_text_absent": "yes",
        "audio_policy_correct": "yes",
        "label_bearing_audio_absent": "yes",
        "required_behavior_audio_preserved": "yes", "artifact_integrity": "yes",
        "manual_description": "A person takes a bag from another.",
        "manual_rationale": "The complete transfer is visible in clean bounds.",
        "final_manual_accept": value,
    }


def test_final_acceptance_is_derived_from_every_atomic_artifact_field(tmp_path):
    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(b"artifact")
    manifest = [{
        "candidate_id": "c", "pillar": "commentary", "uid": "u",
        "artifact_path": str(artifact),
        "artifact_sha256": hashlib.sha256(b"artifact").hexdigest(),
        "behavior_label": "one person takes a bag",
        "approval_status": "awaiting_post_transform_manual_audit",
    }]
    report = validate(manifest, [review()])
    assert report["accepted"] == 1
    assert report["automatic_acceptance"] is False
    bad = review("no")
    bad["final_manual_accept"] = "yes"
    with pytest.raises(ValueError, match="contradicts"):
        validate(manifest, [bad])


def test_final_acceptance_rejects_modified_artifact(tmp_path):
    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(b"modified")
    manifest = [{
        "candidate_id": "c", "pillar": "commentary", "uid": "u",
        "artifact_path": str(artifact), "artifact_sha256": "wrong",
        "behavior_label": "act",
        "approval_status": "awaiting_post_transform_manual_audit",
    }]
    with pytest.raises(ValueError, match="hash-mismatched"):
        validate(manifest, [review()])


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg unavailable")
def test_real_render_is_muted_and_matches_reviewed_duration(tmp_path):
    source = tmp_path / "source.mp4"
    target = tmp_path / "target.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=320x240:r=25:d=3",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
    ], check=True)
    value = plan()
    value.update({
        "source_path": str(source), "proposed_start_sec": 0.5,
        "proposed_end_sec": 2.5,
    })
    subprocess.run(ffmpeg_argv("ffmpeg", value, target), check=True)
    media = probe("ffprobe", target)
    assert media["has_video"] is True
    assert media["has_audio"] is False
    assert abs(media["duration_sec"] - 2.0) <= 0.15


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg unavailable")
def test_real_instructional_render_can_preserve_behavior_audio(tmp_path):
    source = tmp_path / "source.mp4"
    target = tmp_path / "target.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=green:s=320x240:r=25:d=3",
        "-f", "lavfi", "-i", "sine=frequency=550:duration=3",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
    ], check=True)
    value = plan()
    value.update({
        "source_path": str(source), "proposed_start_sec": 0.5,
        "proposed_end_sec": 2.5,
        "transform_required": "temporal_cut_preserve_audio",
    })
    validate_plan([value])
    subprocess.run(ffmpeg_argv("ffmpeg", value, target), check=True)
    media = probe("ffprobe", target)
    assert media["has_video"] is True
    assert media["has_audio"] is True
    assert abs(media["duration_sec"] - 2.0) <= 0.15
