#!/usr/bin/env python3
"""Build a compact image-plus-audio local-review bundle from the fresh audit.

Source videos and rendered MP4s remain on sk3. Storyboards/pages are hash-
verified, resized only when necessary, and re-encoded as review JPEGs. Witnessed
source audio is retained as low-bitrate audio-only review tracks. The portable
manifests retain original and transported hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2

try:
    from scripts.filter_witnessed_scores_to_selection import (
        filter_scores,
        read_jsonl,
        selected_ids,
    )
except ModuleNotFoundError:
    from filter_witnessed_scores_to_selection import (  # type: ignore[no-redef]
        filter_scores,
        read_jsonl,
        selected_ids,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve(path_value: str, manifest: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else manifest.parent / path


def transcode(
    source: Path, target: Path, expected_sha256: str, maximum_width: int, quality: int
) -> dict[str, Any]:
    if not source.is_file() or sha256(source) != expected_sha256:
        raise ValueError(f"missing or hash-mismatched review image: {source}")
    image = cv2.imread(str(source))
    if image is None:
        raise ValueError(f"unreadable review image: {source}")
    original_height, original_width = image.shape[:2]
    if original_width > maximum_width:
        height = max(1, round(original_height * maximum_width / original_width))
        image = cv2.resize(image, (maximum_width, height), interpolation=cv2.INTER_AREA)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), image, [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise OSError(f"failed to write {target}")
    return {
        "path": str(target),
        "sha256": sha256(target),
        "original_sha256": expected_sha256,
        "original_width": original_width,
        "original_height": original_height,
        "review_width": image.shape[1],
        "review_height": image.shape[0],
    }


def extract_review_audio(
    source: Path, target: Path, expected_sha256: str, ffmpeg: str, ffprobe: str,
    start_sec: float | None = None, end_sec: float | None = None,
) -> dict[str, Any]:
    """Create a compact listening track from hash-verified blind audit media."""
    if not source.is_file() or sha256(source) != expected_sha256:
        raise ValueError(f"missing or hash-mismatched audit media: {source}")
    source_media = probe_streams(ffprobe, source)
    if not source_media["has_audio"]:
        raise ValueError(f"source has no audio stream: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-i", str(source),
    ]
    if start_sec is not None or end_sec is not None:
        if start_sec is None or end_sec is None or not 0 <= start_sec < end_sec:
            raise ValueError("review audio requires valid paired temporal bounds")
        argv.extend(["-ss", f"{start_sec:.3f}", "-t", f"{end_sec - start_sec:.3f}"])
    argv.extend([
        "-map", "0:a:0", "-vn", "-sn", "-dn",
        "-ac", "1", "-ar", "16000", "-c:a", "libopus", "-b:a", "32k",
        str(target),
    ])
    subprocess.run(argv, check=True, capture_output=True, text=True, timeout=120)
    payload = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries",
        "format=duration:stream=codec_type", "-of", "json", str(target),
    ], text=True, timeout=60))
    streams = payload.get("streams") or []
    duration = float((payload.get("format") or {}).get("duration") or 0)
    if duration <= 0 or not any(row.get("codec_type") == "audio" for row in streams):
        raise ValueError(f"invalid witnessed review audio: {target}")
    if any(row.get("codec_type") == "video" for row in streams):
        raise ValueError(f"review audio unexpectedly contains video: {target}")
    expected_duration = (
        end_sec - start_sec
        if start_sec is not None and end_sec is not None
        else source_media["duration_sec"]
    )
    tolerance = max(0.25, expected_duration * 0.02)
    if abs(duration - expected_duration) > tolerance:
        raise ValueError(
            f"review audio duration {duration:.3f}s differs from expected "
            f"{expected_duration:.3f}s"
        )
    return {
        "path": str(target),
        "sha256": sha256(target),
        "duration_sec": duration,
        "expected_duration_sec": expected_duration,
        "bytes": target.stat().st_size,
    }


def probe_streams(ffprobe: str, path: Path) -> dict[str, Any]:
    payload = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-show_entries",
        "format=duration:stream=codec_type", "-of", "json", str(path),
    ], text=True, timeout=60))
    streams = payload.get("streams") or []
    return {
        "duration_sec": float((payload.get("format") or {}).get("duration") or 0),
        "has_audio": any(row.get("codec_type") == "audio" for row in streams),
        "has_video": any(row.get("codec_type") == "video" for row in streams),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def copy_metadata(source: Path, target: Path) -> bool:
    if not source.is_file():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def require_file(path: Path) -> Path:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"required completed-batch artifact is missing: {path}")
    return path


def require_unique_ids(
    rows: list[dict[str, Any]], field: str, label: str,
) -> list[str]:
    values = [str(row.get(field) or "") for row in rows]
    if not values or any(not value for value in values):
        raise ValueError(f"{label} has missing {field}")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} has duplicate {field}")
    return values


def require_exact_successful_outputs(
    path: Path,
    expected_ids: list[str],
    id_field: str,
    label: str,
) -> list[dict[str, Any]]:
    """Reject partial, failed, duplicated, or out-of-cohort model logs."""
    rows = read_jsonl(require_file(path))
    ids = require_unique_ids(rows, id_field, label)
    failures = [value for value in rows if value.get("error") not in (None, "")]
    if failures:
        raise ValueError(f"{label} contains {len(failures)} failed outputs")
    if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
        missing = len(set(expected_ids) - set(ids))
        extra = len(set(ids) - set(expected_ids))
        raise ValueError(
            f"{label} cohort mismatch: expected={len(expected_ids)} "
            f"actual={len(ids)} missing={missing} extra={extra}"
        )
    return rows


def _package_into(
    stage: Path, run: Path, v3_scores: Path, out: Path,
    maximum_width: int = 2048, quality: int = 86,
    ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    image_counts = {"instructional": 0, "witnessed": 0, "commentary_masked": 0, "commentary_unmasked": 0}

    instruction_selection_path = require_file(
        stage / "outputs/instructional_v4/sealed_selection.jsonl"
    )
    instruction_selection = read_jsonl(instruction_selection_path)
    instruction_expected_ids = require_unique_ids(
        instruction_selection, "item_id", "instructional selection"
    )
    instruction_manifest = require_file(run / "instructional/storyboard_manifest.jsonl")
    instruction_blind_path = require_file(
        stage / "outputs/instructional_v4/blind_source_manifest.jsonl"
    )
    instruction_blind = read_jsonl(instruction_blind_path)
    instruction_sources = {
        str(row.get("item_id") or ""): row for row in instruction_blind
    }
    if (
        len(instruction_sources) != len(instruction_blind)
        or set(instruction_sources) != set(instruction_expected_ids)
    ):
        raise ValueError("instructional blind sources differ from sealed selection")
    instruction_rows = read_jsonl(instruction_manifest)
    instruction_storyboard_ids = require_unique_ids(
        instruction_rows, "item_id", "instructional storyboards"
    )
    if set(instruction_storyboard_ids) != set(instruction_expected_ids):
        raise ValueError("instructional storyboard cohort differs from sealed selection")
    require_exact_successful_outputs(
        run / "instructional/qwen_temporal_critic_v4.jsonl",
        instruction_expected_ids,
        "item_id",
        "instructional V4 outputs",
    )
    portable_instruction = []
    instructional_audio_count = 0
    instructional_audio_bytes = 0
    for row in instruction_rows:
        if row.get("error"):
            portable_instruction.append(row)
            continue
        target = out / "images/instructional" / f"{int(row['storyboard_index']):04d}.jpg"
        image = transcode(
            resolve(row["sheet_path"], instruction_manifest), target,
            row["sheet_sha256"], maximum_width, quality,
        )
        source_row = instruction_sources[str(row["item_id"])]
        source = resolve(source_row["source_clip"], instruction_blind_path)
        source_hash = str(source_row.get("source_clip_sha256") or "")
        if not source.is_file() or len(source_hash) != 64 or sha256(source) != source_hash:
            raise ValueError(f"{row['item_id']}: missing or hash-mismatched blind source")
        streams = probe_streams(ffprobe, source)
        if not streams["has_video"] or streams["duration_sec"] <= 0:
            raise ValueError(f"{row['item_id']}: invalid blind source media")
        portable = {
            **row, "remote_sheet_path": row["sheet_path"],
            "sheet_path": str(target.relative_to(out)),
            "sheet_sha256": image["sha256"],
            "source_sheet_sha256": image["original_sha256"],
            "remote_source_clip": str(source),
            "source_clip_sha256": source_hash,
            "audio_present": streams["has_audio"],
        }
        if streams["has_audio"]:
            audio_target = out / "audio/instructional" / (
                f"{int(row['storyboard_index']):04d}.ogg"
            )
            audio = extract_review_audio(
                source, audio_target, source_hash, ffmpeg, ffprobe
            )
            portable.update({
                "review_audio_path": str(audio_target.relative_to(out)),
                "review_audio_sha256": audio["sha256"],
                "review_audio_duration_sec": audio["duration_sec"],
                "review_audio_expected_duration_sec": audio["expected_duration_sec"],
            })
            instructional_audio_count += 1
            instructional_audio_bytes += int(audio["bytes"])
        else:
            portable.update({
                "review_audio_path": None, "review_audio_sha256": None,
                "review_audio_duration_sec": None,
                "review_audio_expected_duration_sec": None,
            })
        portable_instruction.append(portable)
        image_counts["instructional"] += 1
    write_jsonl(out / "metadata/instructional/storyboard_manifest.jsonl", portable_instruction)

    witnessed_selection_path = require_file(
        stage / "outputs/witnessed_v6/sealed_selection.jsonl"
    )
    selection = read_jsonl(witnessed_selection_path)
    witnessed_expected_ids = selected_ids(selection)
    witnessed_manifest = require_file(run / "witnessed/manual_media_manifest.jsonl")
    witnessed_rows = read_jsonl(witnessed_manifest)
    witnessed_media_ids = require_unique_ids(
        witnessed_rows, "candidate_id", "witnessed manual media"
    )
    if set(witnessed_media_ids) != set(witnessed_expected_ids):
        raise ValueError("witnessed media cohort differs from sealed selection")
    witnessed_v6_manifest = read_jsonl(require_file(
        run / "witnessed/v6_model_manifest.jsonl"
    ))
    witnessed_v6_ids = require_unique_ids(
        witnessed_v6_manifest, "candidate_id", "witnessed V6 manifest"
    )
    if set(witnessed_v6_ids) != set(witnessed_expected_ids):
        raise ValueError("witnessed V6 manifest differs from sealed selection")
    require_exact_successful_outputs(
        run / "witnessed/qwen_role_causal_binding_v6.jsonl",
        witnessed_expected_ids,
        "candidate_id",
        "witnessed V6 outputs",
    )
    portable_witnessed = []
    witnessed_audio_count = 0
    witnessed_audio_bytes = 0
    for row in witnessed_rows:
        target = out / "images/witnessed" / f"{int(row['audit_candidate_index']):04d}.jpg"
        image = transcode(
            resolve(row["storyboard_path"], witnessed_manifest), target,
            row["storyboard_sha256"], maximum_width, quality,
        )
        value = dict(row)
        value["remote_manual_media_path"] = value.pop("manual_media_path", None)
        value["remote_storyboard_path"] = value["storyboard_path"]
        value["storyboard_path"] = str(target.relative_to(out))
        value["storyboard_sha256"] = image["sha256"]
        value["source_storyboard_sha256"] = image["original_sha256"]
        if row.get("audio_present") is True:
            audio_target = out / "audio/witnessed" / (
                f"{int(row['audit_candidate_index']):04d}.ogg"
            )
            audio = extract_review_audio(
                resolve(row["manual_media_path"], witnessed_manifest),
                audio_target,
                str(row.get("manual_media_sha256") or ""),
                ffmpeg,
                ffprobe,
            )
            value["review_audio_path"] = str(audio_target.relative_to(out))
            value["review_audio_sha256"] = audio["sha256"]
            value["review_audio_duration_sec"] = audio["duration_sec"]
            value["review_audio_expected_duration_sec"] = audio["expected_duration_sec"]
            witnessed_audio_count += 1
            witnessed_audio_bytes += int(audio["bytes"])
        else:
            value["review_audio_path"] = None
            value["review_audio_sha256"] = None
            value["review_audio_duration_sec"] = None
            value["review_audio_expected_duration_sec"] = None
        portable_witnessed.append(value)
        image_counts["witnessed"] += 1
    write_jsonl(out / "metadata/witnessed/manual_media_manifest.jsonl", portable_witnessed)

    commentary_selected_path = require_file(run / "commentary_v2/selected_windows.jsonl")
    commentary_selected = read_jsonl(commentary_selected_path)
    commentary_expected_ids = require_unique_ids(
        commentary_selected, "candidate_id", "commentary selected windows"
    )
    commentary_selection_index = {
        str(row["candidate_id"]): row for row in commentary_selected
    }
    commentary_manifest = require_file(
        run / "commentary_v2/rendered/sealed_render_manifest.jsonl"
    )
    portable_commentary = []
    commentary_rows = read_jsonl(commentary_manifest)
    commentary_render_ids = require_unique_ids(
        commentary_rows, "candidate_id", "commentary render manifest"
    )
    if set(commentary_render_ids) != set(commentary_expected_ids):
        raise ValueError("commentary rendered cohort differs from selected windows")
    require_exact_successful_outputs(
        run / "commentary_v2/qwen_temporal_core_event_v3.jsonl",
        commentary_expected_ids,
        "candidate_id",
        "commentary V3 outputs",
    )
    commentary_audio_count = 0
    commentary_audio_bytes = 0
    commentary_source_hashes: dict[str, str] = {}
    commentary_source_streams: dict[str, dict[str, Any]] = {}
    for row in commentary_rows:
        value = dict(row)
        for kind, path_field, hash_field in (
            ("masked", "page_paths", "page_sha256"),
            ("unmasked", "human_unmasked_page_paths", "human_unmasked_page_sha256"),
        ):
            paths = row.get(path_field) or []
            hashes = row.get(hash_field) or []
            if not paths or len(paths) != len(hashes):
                raise ValueError(
                    f"{row['candidate_id']}: missing/misaligned {kind} review pages"
                )
            portable_paths, portable_hashes, source_hashes = [], [], []
            for page_index, (path_value, expected) in enumerate(zip(paths, hashes)):
                target = out / f"images/commentary_{kind}" / (
                    f"{int(row['audit_index']):04d}_{page_index:02d}.jpg"
                )
                image = transcode(
                    resolve(path_value, commentary_manifest), target,
                    expected, maximum_width, quality,
                )
                portable_paths.append(str(target.relative_to(out)))
                portable_hashes.append(image["sha256"])
                source_hashes.append(image["original_sha256"])
                image_counts[f"commentary_{kind}"] += 1
            value[f"remote_{path_field}"] = paths
            value[path_field] = portable_paths
            value[hash_field] = portable_hashes
            value[f"source_{hash_field}"] = source_hashes
        selection_row = commentary_selection_index[str(row["candidate_id"])]
        source = Path(str(selection_row.get("source_path") or ""))
        if not source.is_file():
            raise ValueError(f"{row['candidate_id']}: missing commentary source media")
        source_key = str(source.resolve())
        if source_key not in commentary_source_hashes:
            commentary_source_hashes[source_key] = sha256(source)
            commentary_source_streams[source_key] = probe_streams(ffprobe, source)
        streams = commentary_source_streams[source_key]
        if not streams["has_video"] or streams["duration_sec"] <= 0:
            raise ValueError(f"{row['candidate_id']}: invalid commentary source media")
        start = float(selection_row["window_start_sec"])
        end = float(selection_row["window_end_sec"])
        if not 0 <= start < end <= streams["duration_sec"] + 0.15:
            raise ValueError(f"{row['candidate_id']}: invalid commentary audio bounds")
        value.update({
            "remote_source_path": source_key,
            "source_media_sha256_observed_at_packaging": commentary_source_hashes[source_key],
            "audio_present": streams["has_audio"],
        })
        if streams["has_audio"]:
            audio_target = out / "audio/commentary" / f"{int(row['audit_index']):04d}.ogg"
            audio = extract_review_audio(
                source, audio_target, commentary_source_hashes[source_key],
                ffmpeg, ffprobe, start, end,
            )
            value.update({
                "review_audio_path": str(audio_target.relative_to(out)),
                "review_audio_sha256": audio["sha256"],
                "review_audio_duration_sec": audio["duration_sec"],
                "review_audio_expected_duration_sec": audio["expected_duration_sec"],
            })
            commentary_audio_count += 1
            commentary_audio_bytes += int(audio["bytes"])
        else:
            value.update({
                "review_audio_path": None, "review_audio_sha256": None,
                "review_audio_duration_sec": None,
                "review_audio_expected_duration_sec": None,
            })
        portable_commentary.append(value)
    write_jsonl(out / "metadata/commentary/sealed_render_manifest.jsonl", portable_commentary)

    # Preserve compact semantic/model/manual artifacts but never MP4s, source
    # links, full-population exports, or all-window baseline tensors.
    files = {
        "stage/instructional_selection.jsonl": stage / "outputs/instructional_v4/sealed_selection.jsonl",
        "stage/instructional_blind_source_manifest.jsonl": (
            stage / "outputs/instructional_v4/blind_source_manifest.jsonl"
        ),
        "stage/instructional_manual_gold.tsv": stage / "outputs/instructional_v4/manual_gold.tsv",
        "stage/witnessed_selection.jsonl": stage / "outputs/witnessed_v6/sealed_selection.jsonl",
        "stage/witnessed_manual_atomic_ledger.tsv": stage / "outputs/witnessed_v6/manual_atomic_ledger.tsv",
        "instructional/qwen_temporal_critic_v4.jsonl": run / "instructional/qwen_temporal_critic_v4.jsonl",
        "witnessed/v6_model_manifest.jsonl": run / "witnessed/v6_model_manifest.jsonl",
        "witnessed/qwen_role_causal_binding_v6.jsonl": run / "witnessed/qwen_role_causal_binding_v6.jsonl",
        "commentary/selected_windows.jsonl": run / "commentary_v2/selected_windows.jsonl",
        "commentary/source_packets.jsonl": run / "commentary_v2/source_packets.jsonl",
        "commentary/video_index.jsonl": run / "commentary_v2/video_index.jsonl",
        "commentary/window_selection_summary.json": run / "commentary_v2/window_selection_summary.json",
        "commentary/blind_manual_review.tsv": run / "commentary_v2/rendered/blind_manual_review.tsv",
        "commentary/post_reveal_manual_review.tsv": run / "commentary_v2/rendered/post_reveal_manual_review.tsv",
        "commentary/qwen_temporal_core_event_v3.jsonl": run / "commentary_v2/qwen_temporal_core_event_v3.jsonl",
        "batch/resume_batch_summary.json": run / "resume_batch_summary.json",
        "batch/cached_audiovisual_model_inventory.json": (
            stage / "cached_audiovisual_model_inventory.json"
        ),
        "batch/fresh_selection_query_provenance_v1.jsonl": (
            stage / "fresh_selection_query_provenance_v1.jsonl"
        ),
        "batch/fresh_selection_query_provenance_v1.summary.json": (
            stage / "fresh_selection_query_provenance_v1.summary.json"
        ),
    }
    copied = []
    for relative, source in files.items():
        require_file(source)
        target = out / "metadata" / relative
        copy_metadata(source, target)
        copied.append({
            "path": str(target.relative_to(out)), "sha256": sha256(target),
            "source_path": str(source),
        })

    v3_slice = filter_scores(selection, read_jsonl(require_file(v3_scores)))
    v3_target = out / "metadata/witnessed/qwen_v3_exact_selection.jsonl"
    write_jsonl(v3_target, v3_slice)
    copied.append({
        "path": str(v3_target.relative_to(out)), "sha256": sha256(v3_target),
        "source_path": str(v3_scores),
    })

    bytes_total = sum(path.stat().st_size for path in out.rglob("*") if path.is_file())
    summary = {
        "kind": "fresh_three_pillar_compact_review_bundle",
        "instructional_items": len(instruction_rows),
        "witnessed_candidates": len(witnessed_rows),
        "commentary_windows": len(portable_commentary),
        "successful_model_output_counts": {
            "instructional": len(instruction_expected_ids),
            "witnessed_v3": len(v3_slice),
            "witnessed_v6": len(witnessed_expected_ids),
            "commentary": len(commentary_expected_ids),
        },
        "review_image_counts": image_counts,
        "instructional_review_audio_count": instructional_audio_count,
        "instructional_review_audio_bytes": instructional_audio_bytes,
        "witnessed_review_audio_count": witnessed_audio_count,
        "witnessed_review_audio_bytes": witnessed_audio_bytes,
        "commentary_review_audio_count": commentary_audio_count,
        "commentary_review_audio_bytes": commentary_audio_bytes,
        "metadata_files": copied,
        "bytes_before_summary": bytes_total,
        "source_videos_copied": False,
        "rendered_mp4s_copied": False,
        "audio_only_review_tracks_included": (
            instructional_audio_count + witnessed_audio_count + commentary_audio_count > 0
        ),
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }
    (out / "bundle_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def package(
    stage: Path, run: Path, v3_scores: Path, out: Path,
    maximum_width: int = 2048, quality: int = 86,
    ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    """Build privately, then atomically publish a complete review bundle."""
    if out.exists():
        raise FileExistsError(out)
    temporary = out.with_name(f".{out.name}.building-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        summary = _package_into(
            stage, run, v3_scores, temporary,
            maximum_width, quality, ffmpeg, ffprobe,
        )
        os.replace(temporary, out)
        return summary
    except BaseException:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--v3-scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--maximum-width", type=int, default=2048)
    parser.add_argument("--jpeg-quality", type=int, default=86)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    print(json.dumps(package(
        args.stage, args.run, args.v3_scores, args.out,
        args.maximum_width, args.jpeg_quality, args.ffmpeg, args.ffprobe,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
