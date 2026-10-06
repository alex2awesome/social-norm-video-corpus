#!/usr/bin/env python3
"""Verify the compact review bundle again after transport."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}
AUDIO_SUFFIXES = {".ogg", ".opus", ".mp3", ".m4a", ".wav", ".flac"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def inside(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        raise ValueError(f"portable bundle path is absolute: {relative}")
    resolved = (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"portable bundle path escapes root: {relative}") from exc
    return resolved


def verify_image(root: Path, relative: str, expected: str) -> None:
    path = inside(root, relative)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing or linked review image: {relative}")
    if sha256(path) != expected:
        raise ValueError(f"review image hash mismatch: {relative}")


def verify_audio(root: Path, relative: str, expected: str) -> int:
    path = inside(root, relative)
    if not path.is_file() or path.is_symlink() or path.suffix.lower() not in AUDIO_SUFFIXES:
        raise ValueError(f"missing, linked, or invalid review audio: {relative}")
    if sha256(path) != expected:
        raise ValueError(f"review audio hash mismatch: {relative}")
    return path.stat().st_size


def verify(root: Path) -> dict[str, Any]:
    summary_path = root / "bundle_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("kind") != "fresh_three_pillar_compact_review_bundle":
        raise ValueError("unexpected bundle kind")
    if summary.get("source_videos_copied") is not False or summary.get("rendered_mp4s_copied") is not False:
        raise ValueError("bundle claims copied source/rendered videos")

    for item in summary.get("metadata_files") or []:
        path = inside(root, str(item.get("path") or ""))
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"missing or linked metadata: {item.get('path')}")
        if sha256(path) != item.get("sha256"):
            raise ValueError(f"metadata hash mismatch: {item.get('path')}")

    instruction = read_jsonl(
        root / "metadata/instructional/storyboard_manifest.jsonl"
    )
    witnessed = read_jsonl(root / "metadata/witnessed/manual_media_manifest.jsonl")
    commentary = read_jsonl(
        root / "metadata/commentary/sealed_render_manifest.jsonl"
    )
    if len(instruction) != int(summary["instructional_items"]):
        raise ValueError("instructional manifest count mismatch")
    if len(witnessed) != int(summary["witnessed_candidates"]):
        raise ValueError("witnessed manifest count mismatch")
    if len(commentary) != int(summary["commentary_windows"]):
        raise ValueError("commentary manifest count mismatch")

    counts = {
        "instructional": 0,
        "witnessed": 0,
        "commentary_masked": 0,
        "commentary_unmasked": 0,
    }
    audio_count = 0
    audio_bytes = 0
    instructional_audio_count = 0
    instructional_audio_bytes = 0
    commentary_audio_count = 0
    commentary_audio_bytes = 0
    referenced_audio: set[str] = set()
    for row in instruction:
        if row.get("error"):
            raise ValueError("instructional storyboard manifest contains an error")
        verify_image(root, row["sheet_path"], row["sheet_sha256"])
        counts["instructional"] += 1
        if row.get("audio_present") is True:
            if not row.get("review_audio_path") or not row.get("review_audio_sha256"):
                raise ValueError(f"{row.get('item_id')}: missing instructional review audio")
            instructional_audio_bytes += verify_audio(
                root, row["review_audio_path"], row["review_audio_sha256"]
            )
            referenced_audio.add(str(row["review_audio_path"]))
            instructional_audio_count += 1
        elif row.get("review_audio_path") or row.get("review_audio_sha256"):
            raise ValueError(f"{row.get('item_id')}: unexpected instructional review audio")
    for row in witnessed:
        verify_image(root, row["storyboard_path"], row["storyboard_sha256"])
        counts["witnessed"] += 1
        if row.get("audio_present") is True:
            if not row.get("review_audio_path") or not row.get("review_audio_sha256"):
                raise ValueError(f"{row.get('candidate_id')}: missing review audio")
            audio_bytes += verify_audio(
                root, row["review_audio_path"], row["review_audio_sha256"]
            )
            referenced_audio.add(str(row["review_audio_path"]))
            audio_count += 1
        elif row.get("review_audio_path") or row.get("review_audio_sha256"):
            raise ValueError(f"{row.get('candidate_id')}: unexpected review audio")
    for row in commentary:
        for count_name, path_field, hash_field in (
            ("commentary_masked", "page_paths", "page_sha256"),
            (
                "commentary_unmasked", "human_unmasked_page_paths",
                "human_unmasked_page_sha256",
            ),
        ):
            paths, hashes = row.get(path_field) or [], row.get(hash_field) or []
            if not paths or len(paths) != len(hashes):
                raise ValueError(f"{row.get('candidate_id')}: incomplete page list")
            for relative, expected in zip(paths, hashes):
                verify_image(root, relative, expected)
                counts[count_name] += 1
        if row.get("audio_present") is True:
            if not row.get("review_audio_path") or not row.get("review_audio_sha256"):
                raise ValueError(f"{row.get('candidate_id')}: missing commentary review audio")
            commentary_audio_bytes += verify_audio(
                root, row["review_audio_path"], row["review_audio_sha256"]
            )
            referenced_audio.add(str(row["review_audio_path"]))
            commentary_audio_count += 1
        elif row.get("review_audio_path") or row.get("review_audio_sha256"):
            raise ValueError(f"{row.get('candidate_id')}: unexpected commentary review audio")
    if counts != summary.get("review_image_counts"):
        raise ValueError("review image counts differ from bundle summary")
    if audio_count != int(summary.get("witnessed_review_audio_count") or 0):
        raise ValueError("review audio count differs from bundle summary")
    if audio_bytes != int(summary.get("witnessed_review_audio_bytes") or 0):
        raise ValueError("review audio bytes differ from bundle summary")
    if instructional_audio_count != int(summary.get("instructional_review_audio_count") or 0):
        raise ValueError("instructional review audio count differs from bundle summary")
    if instructional_audio_bytes != int(summary.get("instructional_review_audio_bytes") or 0):
        raise ValueError("instructional review audio bytes differ from bundle summary")
    if commentary_audio_count != int(summary.get("commentary_review_audio_count") or 0):
        raise ValueError("commentary review audio count differs from bundle summary")
    if commentary_audio_bytes != int(summary.get("commentary_review_audio_bytes") or 0):
        raise ValueError("commentary review audio bytes differ from bundle summary")

    files = [path for path in root.rglob("*") if path.is_file()]
    links = [path for path in root.rglob("*") if path.is_symlink()]
    videos = [path for path in files if path.suffix.lower() in VIDEO_SUFFIXES]
    audio_files = {
        str(path.relative_to(root))
        for path in files if path.suffix.lower() in AUDIO_SUFFIXES
    }
    if links:
        raise ValueError(f"bundle contains {len(links)} symlinks")
    if videos:
        raise ValueError(f"bundle contains {len(videos)} video files")
    if audio_files != referenced_audio:
        raise ValueError("bundle contains missing or unreferenced audio files")
    return {
        "kind": "fresh_three_pillar_compact_review_bundle_verification",
        "files": len(files),
        "bytes": sum(path.stat().st_size for path in files),
        "review_image_counts": counts,
        "instructional_review_audio_count": instructional_audio_count,
        "instructional_review_audio_bytes": instructional_audio_bytes,
        "commentary_review_audio_count": commentary_audio_count,
        "commentary_review_audio_bytes": commentary_audio_bytes,
        "witnessed_review_audio_count": audio_count,
        "witnessed_review_audio_bytes": audio_bytes,
        "metadata_files_verified": len(summary.get("metadata_files") or []),
        "source_videos_present": False,
        "symlinks_present": False,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = verify(args.bundle)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
