import hashlib

from scripts.validate_witnessed_video_asr_corpus_manifest import validate


def row(tmp_path, candidate_id="a"):
    media = tmp_path / f"{candidate_id}.mp4"
    media.write_bytes(b"video")
    return {
        "candidate_id": candidate_id,
        "item_id": "witnessed:youtube__a:0",
        "uid": "youtube__a",
        "candidate_video_path": str(media),
        "candidate_video_sha256": hashlib.sha256(b"video").hexdigest(),
        "candidate_relative_start_sec": 1,
        "candidate_relative_end_sec": 2,
        "window_duration_sec": 3,
        "context_before": [],
        "context_after": ["stop"],
        "two_stage_full_proxy_then_candidate_cut": True,
        "max_source_frames": 96,
        "max_width": 512,
        "sampling_fps": 2.0,
        "error": None,
    }


def test_validate_passes_exact_contract_and_hash(tmp_path):
    report = validate(
        [row(tmp_path)], tmp_path, expected_candidates=1, hash_sample=1, seed="s"
    )
    assert report["passed"] is True
    assert report["context_after_nonempty_fraction"] == 1


def test_validate_reports_count_contract_timing_and_hash_failures(tmp_path):
    broken = row(tmp_path)
    broken["max_source_frames"] = 95
    broken["candidate_relative_end_sec"] = 5
    broken["candidate_video_sha256"] = "bad"
    report = validate(
        [broken], tmp_path, expected_candidates=2, hash_sample=1, seed="s"
    )
    kinds = {issue["kind"] for issue in report["issues"]}
    assert report["passed"] is False
    assert kinds == {
        "candidate_count_mismatch",
        "representation_contract_error",
        "candidate_timing_error",
        "sampled_media_hash_mismatch",
    }
