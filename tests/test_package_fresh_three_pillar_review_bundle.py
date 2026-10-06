from pathlib import Path

import cv2
import numpy as np

import json

import pytest

from scripts.package_fresh_three_pillar_review_bundle import (
    extract_review_audio,
    package,
    sha256,
    transcode,
)
from scripts.verify_fresh_three_pillar_review_bundle import verify


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def review_image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((60, 120, 3), dtype=np.uint8))
    return sha256(path)


def build_complete_batch(tmp_path: Path) -> tuple[Path, Path, Path]:
    stage = tmp_path / "stage"
    run = stage / "run"
    instruction_selection = [{"item_id": "i1"}]
    witnessed_selection = [{
        "item_id": "clip1",
        "candidates": [{"candidate_id": "w1"}],
    }]
    write_jsonl(
        stage / "outputs/instructional_v4/sealed_selection.jsonl",
        instruction_selection,
    )
    instruction_video = stage / "outputs/instructional_v4/blind_source.mp4"
    instruction_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(instruction_video), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64)
    )
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()
    write_jsonl(
        stage / "outputs/instructional_v4/blind_source_manifest.jsonl",
        [{
            "item_id": "i1", "uid": "u1", "pillar": "instructional",
            "source_clip": str(instruction_video),
            "source_clip_sha256": sha256(instruction_video),
        }],
    )
    write_jsonl(
        stage / "outputs/witnessed_v6/sealed_selection.jsonl",
        witnessed_selection,
    )
    for path in (
        stage / "outputs/instructional_v4/manual_gold.tsv",
        stage / "outputs/witnessed_v6/manual_atomic_ledger.tsv",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("header\n")

    instruction_image = run / "instructional/rendered/i.jpg"
    write_jsonl(run / "instructional/storyboard_manifest.jsonl", [{
        "item_id": "i1",
        "storyboard_index": 0,
        "sheet_path": str(instruction_image),
        "sheet_sha256": review_image(instruction_image),
    }])
    write_jsonl(run / "instructional/qwen_temporal_critic_v4.jsonl", [{
        "item_id": "i1", "result": {"demo_candidate": True}, "error": None,
    }])

    witnessed_image = run / "witnessed/storyboards/w.jpg"
    write_jsonl(run / "witnessed/manual_media_manifest.jsonl", [{
        "candidate_id": "w1",
        "audit_candidate_index": 0,
        "storyboard_path": str(witnessed_image),
        "storyboard_sha256": review_image(witnessed_image),
        "manual_media_path": "/remote/do-not-copy.mp4",
    }])
    write_jsonl(run / "witnessed/v6_model_manifest.jsonl", [{
        "candidate_id": "w1", "item_id": "clip1",
    }])
    write_jsonl(run / "witnessed/qwen_role_causal_binding_v6.jsonl", [{
        "candidate_id": "w1", "result": {"strict_pass": False}, "error": None,
    }])

    selected = [{
        "candidate_id": "c1", "window_id": "c1", "uid": "u1",
        "source_path": str(instruction_video),
        "window_start_sec": 0.0, "window_end_sec": 0.8,
    }]
    write_jsonl(run / "commentary_v2/selected_windows.jsonl", selected)
    commentary_manifest = run / "commentary_v2/rendered/sealed_render_manifest.jsonl"
    masked = commentary_manifest.parent / "rendered/masked.jpg"
    unmasked = commentary_manifest.parent / "rendered/unmasked.jpg"
    write_jsonl(commentary_manifest, [{
        "audit_index": 0,
        "candidate_id": "c1",
        "page_paths": [str(masked)],
        "page_sha256": [review_image(masked)],
        "human_unmasked_page_paths": [str(unmasked)],
        "human_unmasked_page_sha256": [review_image(unmasked)],
    }])
    write_jsonl(run / "commentary_v2/qwen_temporal_core_event_v3.jsonl", [{
        "candidate_id": "c1", "stage_a": {}, "stage_b": {}, "error": None,
    }])
    write_jsonl(run / "commentary_v2/source_packets.jsonl", [{"item_id": "s1"}])
    write_jsonl(run / "commentary_v2/video_index.jsonl", [{"uid": "u1"}])
    (run / "commentary_v2/window_selection_summary.json").write_text("{}\n")
    for name in ("blind_manual_review.tsv", "post_reveal_manual_review.tsv"):
        (run / "commentary_v2/rendered" / name).write_text("header\n")
    (run / "resume_batch_summary.json").write_text('{"exit_status": 0}\n')
    (stage / "cached_audiovisual_model_inventory.json").write_text(
        '{"candidate_count": 0}\n'
    )
    write_jsonl(stage / "fresh_selection_query_provenance_v1.jsonl", [{"uid": "u"}])
    (stage / "fresh_selection_query_provenance_v1.summary.json").write_text(
        '{"items": 1}\n'
    )

    v3_scores = stage / "v3.jsonl"
    write_jsonl(v3_scores, [{
        "candidate_id": "w1", "result": {"strict_pass": False}, "error": None,
    }])
    return stage, run, v3_scores


def test_transcode_verifies_hash_and_bounds_width(tmp_path: Path):
    source = tmp_path / "source.jpg"
    target = tmp_path / "out" / "review.jpg"
    assert cv2.imwrite(str(source), np.zeros((100, 400, 3), dtype=np.uint8))
    result = transcode(source, target, sha256(source), 200, 80)
    assert result["review_width"] == 200
    assert result["review_height"] == 50
    assert target.is_file()


def test_transcode_rejects_hash_mismatch(tmp_path: Path):
    source = tmp_path / "source.jpg"
    assert cv2.imwrite(str(source), np.zeros((20, 20, 3), dtype=np.uint8))
    try:
        transcode(source, tmp_path / "out.jpg", "0" * 64, 200, 80)
    except ValueError as exc:
        assert "hash-mismatched" in str(exc)
    else:
        raise AssertionError("hash mismatch should fail")


def test_package_requires_complete_exact_cohorts_and_copies_no_video(tmp_path: Path):
    stage, run, v3_scores = build_complete_batch(tmp_path)
    out = tmp_path / "bundle"
    summary = package(stage, run, v3_scores, out, maximum_width=80, quality=80)
    assert summary["instructional_items"] == 1
    assert summary["witnessed_candidates"] == 1
    assert summary["commentary_windows"] == 1
    assert summary["successful_model_output_counts"] == {
        "instructional": 1,
        "witnessed_v3": 1,
        "witnessed_v6": 1,
        "commentary": 1,
    }
    assert len(list(out.rglob("*.jpg"))) == 4
    assert not list(out.rglob("*.mp4"))
    verification = verify(out)
    assert verification["review_image_counts"] == summary["review_image_counts"]
    assert verification["source_videos_present"] is False


def test_package_rejects_partial_model_output_before_review_transfer(tmp_path: Path):
    stage, run, v3_scores = build_complete_batch(tmp_path)
    (run / "witnessed/qwen_role_causal_binding_v6.jsonl").write_text("")
    with pytest.raises(FileNotFoundError, match="required completed-batch artifact"):
        package(stage, run, v3_scores, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".bundle.building-*"))


def test_extract_review_audio_is_small_hash_verified_and_video_free(tmp_path: Path):
    import shutil
    import subprocess
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg unavailable")
    source = tmp_path / "source.mp4"
    target = tmp_path / "review.ogg"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=160x120:r=10:d=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(source),
    ], check=True)
    result = extract_review_audio(
        source, target, sha256(source), "ffmpeg", "ffprobe"
    )
    assert result["duration_sec"] >= 1.9
    assert abs(result["duration_sec"] - result["expected_duration_sec"]) <= 0.25
    assert result["sha256"] == sha256(target)
    assert result["bytes"] < source.stat().st_size


def test_package_and_verifier_cover_witnessed_audio_tracks(tmp_path: Path):
    import shutil
    import subprocess
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg unavailable")
    stage, run, v3_scores = build_complete_batch(tmp_path)
    media = run / "witnessed/manual_media.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=black:s=160x120:r=10:d=1",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(media),
    ], check=True)
    manifest = run / "witnessed/manual_media_manifest.jsonl"
    row = json.loads(manifest.read_text())
    row.update({
        "manual_media_path": str(media),
        "manual_media_sha256": sha256(media),
        "audio_present": True,
    })
    write_jsonl(manifest, [row])
    out = tmp_path / "bundle"
    summary = package(stage, run, v3_scores, out)
    assert summary["witnessed_review_audio_count"] == 1
    assert summary["witnessed_review_audio_bytes"] > 0
    report = verify(out)
    assert report["witnessed_review_audio_count"] == 1
    assert len(list(out.rglob("*.ogg"))) == 1


def test_package_and_verifier_cover_instructional_speech_audio(tmp_path: Path):
    import shutil
    import subprocess
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg unavailable")
    stage, run, v3_scores = build_complete_batch(tmp_path)
    media = stage / "outputs/instructional_v4/instructional_audio.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:r=10:d=1",
        "-f", "lavfi", "-i", "sine=frequency=550:duration=1",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(media),
    ], check=True)
    manifest = stage / "outputs/instructional_v4/blind_source_manifest.jsonl"
    row = json.loads(manifest.read_text())
    row.update({"source_clip": str(media), "source_clip_sha256": sha256(media)})
    write_jsonl(manifest, [row])
    out = tmp_path / "bundle"
    summary = package(stage, run, v3_scores, out)
    assert summary["instructional_review_audio_count"] == 1
    report = verify(out)
    assert report["instructional_review_audio_count"] == 1
    assert len(list((out / "audio/instructional").glob("*.ogg"))) == 1


def test_package_and_verifier_cover_commentary_window_audio(tmp_path: Path):
    import shutil
    import subprocess
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("ffmpeg unavailable")
    stage, run, v3_scores = build_complete_batch(tmp_path)
    media = run / "commentary_v2/source_audio.mp4"
    subprocess.run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-f", "lavfi", "-i", "color=c=yellow:s=160x120:r=10:d=2",
        "-f", "lavfi", "-i", "sine=frequency=660:duration=2",
        "-c:v", "libx264", "-c:a", "aac", "-shortest", str(media),
    ], check=True)
    selected_path = run / "commentary_v2/selected_windows.jsonl"
    row = json.loads(selected_path.read_text())
    row.update({
        "source_path": str(media), "window_start_sec": 0.4,
        "window_end_sec": 1.6,
    })
    write_jsonl(selected_path, [row])
    out = tmp_path / "bundle"
    summary = package(stage, run, v3_scores, out)
    assert summary["commentary_review_audio_count"] == 1
    report = verify(out)
    assert report["commentary_review_audio_count"] == 1
    assert len(list((out / "audio/commentary").glob("*.ogg"))) == 1
