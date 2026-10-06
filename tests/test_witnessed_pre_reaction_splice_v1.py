import copy
import shutil
import subprocess

import pytest

from scripts.build_witnessed_pre_reaction_splice_plans_v1 import build, validate
from scripts.materialize_witnessed_pre_reaction_splices_v1 import ffmpeg_argv, probe_media
from scripts.validate_witnessed_pre_reaction_artifacts_v1 import validate as validate_artifact


def atomic():
    return {
        "candidate_id": "c", "reaction_grounded": "yes",
        "action_before_or_overlaps_response": "yes",
        "response_targets_action": "yes", "responder_role": "separate_bystander",
        "response_content": "targeted_objection",
        "trigger_kind": "interpersonal_treatment",
        "staging": "no_clear_staging_evidence",
    }


def test_builds_boundary_review_only_for_strict_manual_reaction_candidate():
    manifest = [{
        "candidate_id": "c", "item_id": "i", "uid": "u",
        "candidate_video_path": "/candidate.mp4", "candidate_video_sha256": "h",
        "window_duration_sec": 8.0, "candidate_relative_start_sec": 5.0,
    }]
    rows = build(manifest, [atomic()])
    assert len(rows) == 1
    assert rows[0]["reaction_start_hint_sec"] == "5.0"
    assert rows[0]["ready_for_splice"] == ""


def filled_boundary():
    return {
        "candidate_id": "c", "item_id": "i", "uid": "u",
        "candidate_video_path": "/candidate.mp4", "candidate_video_sha256": "h",
        "window_duration_sec": "8", "reaction_start_hint_sec": "5",
        "trigger_action_label": "one person shoves another",
        "action_start_sec": "1", "action_end_sec": "4.7", "reaction_start_sec": "5",
        "pre_reaction_action_complete": "yes", "reaction_signal_before_cut": "no",
        "exact_behavior_label_supported": "yes", "start_boundary_clean": "yes",
        "end_boundary_clean": "yes", "audiovisual_boundary_reviewed": "yes",
        "manual_rationale": "The shove completes before a third party objects.",
        "ready_for_splice": "yes",
    }


def test_boundary_seal_requires_safety_margin_and_every_atom():
    plan = validate([filled_boundary()])[0]
    assert plan["ready_for_splice"] is True
    argv = ffmpeg_argv("ffmpeg", plan, __import__("pathlib").Path("out.mp4"))
    assert "-an" not in argv
    assert argv.index("-i") < argv.index("-ss")
    assert "0:a:0?" in argv
    bad = filled_boundary(); bad["action_end_sec"] = "4.95"
    with pytest.raises(ValueError, match="100ms"):
        validate([bad])


def test_boundary_seal_requires_complete_atomic_review():
    bad = filled_boundary(); bad["end_boundary_clean"] = "uncertain"
    with pytest.raises(ValueError, match="incomplete atomic"):
        validate([bad])


def post_review():
    return {
        "candidate_id": "c", "uid": "u", "social_action_complete": "yes",
        "exact_behavior_label_supported": "yes", "reaction_visible_absent": "yes",
        "reaction_audible_absent": "yes", "start_boundary_clean": "yes",
        "end_boundary_clean": "yes", "artifact_integrity": "yes",
        "manual_description": "A person shoves another.",
        "manual_rationale": "The shove is complete and the objection is absent.",
        "final_manual_accept": "yes",
    }


def test_final_witnessed_accept_requires_audible_reaction_absence(tmp_path):
    artifact = tmp_path / "a.mp4"
    artifact.write_bytes(b"artifact")
    import hashlib
    manifest = [{
        "candidate_id": "c", "uid": "u", "artifact_path": str(artifact),
        "artifact_sha256": hashlib.sha256(b"artifact").hexdigest(),
        "behavior_label": "one person shoves another",
        "approval_status": "awaiting_post_splice_manual_audit",
    }]
    assert validate_artifact(manifest, [post_review()])["accepted"] == 1
    bad = copy.deepcopy(post_review())
    bad["reaction_audible_absent"] = "no"
    with pytest.raises(ValueError, match="inconsistent"):
        validate_artifact(manifest, [bad])


def test_final_witnessed_accept_rejects_changed_artifact(tmp_path):
    artifact = tmp_path / "a.mp4"
    artifact.write_bytes(b"changed")
    manifest = [{
        "candidate_id": "c", "uid": "u", "artifact_path": str(artifact),
        "artifact_sha256": "wrong", "behavior_label": "shove",
        "approval_status": "awaiting_post_splice_manual_audit",
    }]
    with pytest.raises(ValueError, match="hash-mismatched"):
        validate_artifact(manifest, [post_review()])


@pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"), reason="ffmpeg unavailable")
def test_real_pre_reaction_render_preserves_audio_and_exact_interval(tmp_path):
    source = tmp_path / "source.mp4"
    target = tmp_path / "target.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=320x240:r=25:d=3",
        "-f", "lavfi", "-i", "sine=frequency=660:duration=3",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
    ], check=True)
    value = validate([filled_boundary()])[0]
    value.update({
        "source_path": str(source), "action_start_sec": 0.5,
        "action_end_sec": 2.5,
    })
    subprocess.run(ffmpeg_argv("ffmpeg", value, target), check=True)
    media = probe_media("ffprobe", target)
    assert media["has_video"] is True
    assert media["has_audio"] is True
    assert abs(media["duration_sec"] - 2.0) <= 0.15
