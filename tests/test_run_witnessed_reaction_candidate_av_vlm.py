import json
import hashlib
import urllib.error
from pathlib import Path

import pytest

from scripts.run_witnessed_reaction_candidate_av_vlm import (
    SYSTEM_PROMPTS,
    parse_result,
    pending_manifest_rows,
    resolve_media,
    retryable_error,
    successful_candidate_ids,
)
from scripts.run_witnessed_reaction_candidate_frames_asr_v7 import (
    SYSTEM_V7_FRAMES_ASR_ROLE_BINDING,
)


def valid():
    return {
        "reaction_grounded": "yes",
        "action_before_or_overlaps_response": "yes",
        "response_targets_action": "yes",
        "responder_role": "separate_bystander",
        "response_content": "protective_intervention",
        "trigger_kind": "interpersonal_treatment",
        "staging": "no_clear_staging_evidence",
        "evidence": "A third person steps between two people after one grabs the other.",
    }


def test_parse_result_accepts_exact_atomic_schema():
    assert parse_result(json.dumps(valid()))["responder_role"] == "separate_bystander"


def test_parse_result_rejects_unknown_trigger():
    row = valid()
    row["trigger_kind"] = "social_norm"
    with pytest.raises(ValueError, match="trigger_kind"):
        parse_result(json.dumps(row))


def test_parse_result_rejects_extra_composite_label():
    row = valid()
    row["is_witnessed"] = "yes"
    with pytest.raises(ValueError, match="schema mismatch"):
        parse_result(json.dumps(row))


def test_resolve_media_supports_storyboard_relative_to_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sheet = tmp_path / "s.jpg"
    sheet.write_bytes(b"image")
    import hashlib
    row = {
        "candidate_id": "a",
        "sheet_path": "s.jpg",
        "sheet_sha256": hashlib.sha256(b"image").hexdigest(),
        "_manifest_dir": str(tmp_path / "other"),
    }
    path, field = resolve_media(row, "image")
    assert path == sheet
    assert field == "sheet_sha256"


def test_resolve_media_repairs_stale_absolute_path_by_hash(tmp_path):
    import hashlib

    media_dir = tmp_path / "media"
    media_dir.mkdir()
    clip = media_dir / "candidate.mp4"
    clip.write_bytes(b"video")
    row = {
        "candidate_id": "a",
        "candidate_video_path": "/afs/inaccessible/media/candidate.mp4",
        "candidate_video_sha256": hashlib.sha256(b"video").hexdigest(),
        "_manifest_dir": str(tmp_path),
    }
    path, field = resolve_media(row, "video")
    assert path == clip
    assert field == "candidate_video_sha256"


def test_stale_path_fallback_still_enforces_hash(tmp_path):
    import hashlib

    media_dir = tmp_path / "media"
    media_dir.mkdir()
    (media_dir / "candidate.mp4").write_bytes(b"wrong")
    row = {
        "candidate_id": "a",
        "candidate_video_path": "/afs/inaccessible/media/candidate.mp4",
        "candidate_video_sha256": hashlib.sha256(b"expected").hexdigest(),
        "_manifest_dir": str(tmp_path),
    }
    with pytest.raises(ValueError, match="hash mismatch"):
        resolve_media(row, "video")


def test_v4_prompt_matches_silent_video_asr_contract():
    prompt = SYSTEM_PROMPTS["v4_silent_video_asr"]
    assert "no audio track" in prompt
    assert "Never claim to hear" in prompt
    assert "ASR alone cannot establish" in prompt
    assert "offscreen_or_unresolved" in prompt


def test_frozen_v3_prompt_remains_available_for_reproducibility():
    assert "Use the video and its audio" in SYSTEM_PROMPTS["v3_audiovisual_wording"]


def test_v5_prompt_requires_explicit_real_audio_presence():
    prompt = " ".join(SYSTEM_PROMPTS["v5_real_audio_asr"].split())
    assert "original source audio is present" in prompt
    assert "When it says audio is absent" in prompt
    assert "do not bind a speaker" in prompt


def test_v6_prompt_preregisters_role_and_causal_failure_controls():
    prompt = " ".join(SYSTEM_PROMPTS["v6_role_causal_binding"].split())
    assert "TRIGGER ACTOR" in prompt
    assert "DIRECT TARGET" in prompt
    assert "RESPONSE ACTOR" in prompt
    assert "PRIOR PARTICIPATION" in prompt
    assert "neither endpoint of the triggering action" in prompt
    assert "ordinary questions" in prompt
    assert "violator's own normative demand" in prompt
    assert "Keep staging independent from reaction presence" in prompt
    assert "accident_or_involuntary" in prompt


def test_v7_prompt_states_actual_qwen3_vl_frame_plus_asr_boundary():
    prompt = " ".join(SYSTEM_V7_FRAMES_ASR_ROLE_BINDING.split())
    assert "does not consume the MP4 audio track" in prompt
    assert "Never claim to hear" in prompt
    assert "cannot by itself bind a speaker" in prompt
    assert "offscreen_or_unresolved" in prompt
    assert "TRIGGER ACTOR" in prompt


def test_v6_fresh_transfer_preregistration_freezes_current_runner():
    root = Path(__file__).resolve().parents[1]
    prereg = json.loads((
        root
        / "audit_runs/20260806_witnessed_role_causal_binding_v6_fresh_transfer/preregistration.json"
    ).read_text())
    runner = root / prereg["runner"]
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == prereg["runner_sha256"]
    assert prereg["freshness"]["human_gold_sealed_before_v6_scoring"] is True
    assert prereg["manual_audit"]["audit_every_selected_candidate"] is True
    assert prereg["evaluation_contract"]["organic_source_routing_reported_separately"] is True


def test_append_only_success_wins_over_earlier_failure():
    outputs = [
        {"candidate_id": "a", "model": "qwen", "error": "timeout", "result": None},
        {"candidate_id": "a", "model": "qwen", "error": None, "result": valid()},
        {"candidate_id": "b", "model": "qwen", "error": "parse", "result": None},
    ]
    assert successful_candidate_ids(outputs, "qwen") == {"a"}
    assert [row["candidate_id"] for row in pending_manifest_rows(
        [{"candidate_id": "a"}, {"candidate_id": "b"}], outputs, "qwen"
    )] == ["b"]


def test_other_model_or_resultless_row_does_not_suppress_retry():
    outputs = [
        {"candidate_id": "a", "model": "other", "error": None, "result": valid()},
        {"candidate_id": "b", "model": "qwen", "error": None, "result": None},
    ]
    pending = pending_manifest_rows(
        [{"candidate_id": "a"}, {"candidate_id": "b"}], outputs, "qwen"
    )
    assert [row["candidate_id"] for row in pending] == ["a", "b"]


@pytest.mark.parametrize("code", [400, 401, 403, 404, 422])
def test_deterministic_http_client_errors_are_not_retried(code):
    error = urllib.error.HTTPError("http://local", code, "bad", {}, None)
    assert retryable_error(error) is False


@pytest.mark.parametrize("code", [408, 409, 425, 429, 500, 503])
def test_transient_http_errors_are_retried(code):
    error = urllib.error.HTTPError("http://local", code, "retry", {}, None)
    assert retryable_error(error) is True


def test_model_schema_and_transport_errors_are_retried():
    assert retryable_error(ValueError("schema")) is True
    assert retryable_error(TimeoutError("slow")) is True
