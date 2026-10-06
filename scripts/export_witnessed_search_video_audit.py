#!/usr/bin/env python3
"""Download a frozen search-shadow sample and render temporal audit sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from export_visual_audit_batch import extract_frames, sha256_json  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_media(media_dir: Path, uid: str) -> list[Path]:
    return [
        path
        for path in media_dir.glob(f"{uid}.*")
        if path.suffix not in {".part", ".ytdl"} and ".unsupported-" not in path.name
    ]


def make_sheet(frame_records: list[dict], frames_dir: Path, target: Path) -> None:
    images = []
    for frame in frame_records:
        image = Image.open(frames_dir / Path(frame["path"]).name).convert("RGB")
        image.thumbnail((384, 240))
        cell = Image.new("RGB", (400, 270), "white")
        cell.paste(image, ((400 - image.width) // 2, 22))
        ImageDraw.Draw(cell).text(
            (6, 5), f"{frame['frame_index']:02d}  t={frame['timestamp']:.1f}s", fill="black"
        )
        images.append(cell)
    columns = 4
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 400, rows * 270), "#dddddd")
    for index, image in enumerate(images):
        sheet.paste(image, ((index % columns) * 400, (index // columns) * 270))
    sheet.save(target, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enumeration", type=Path, action="append", required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--yt-dlp", required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    parser.add_argument("--max-frames", type=int, default=16)
    parser.add_argument(
        "--format",
        default="bv*[vcodec^=avc1]+ba/b[vcodec^=avc1]/bv*+ba/b",
        help="Prefer H.264 because the audit ffmpeg build may not decode AV1.",
    )
    parser.add_argument("--concurrent-fragments", type=int, default=4)
    args = parser.parse_args()

    enumerations = [json.loads(path.read_text()) for path in args.enumeration]
    selection = json.loads(args.selection.read_text())
    allowed_title_skips = set(
        selection.get("audit_only_allow_title_skipped_uids", [])
    )
    selected_uids = {item["uid"] for item in selection["items"]}
    if not allowed_title_skips.issubset(selected_uids):
        raise SystemExit("title-skip override contains a UID absent from selection")
    lookup = {}
    lookup_without_group = {}
    for enumeration in enumerations:
        for row in enumeration["records"]:
            if row.get("enumeration_summary"):
                continue
            lookup[(row.get("group"), row.get("query"), row.get("uid"))] = row
            key = (row.get("query"), row.get("uid"))
            if key in lookup_without_group:
                raise SystemExit(f"ambiguous query/UID across enumerations: {key}")
            lookup_without_group[key] = row
    if len(selection["items"]) != len(
        {(item.get("pillar"), item.get("group"), item["uid"]) for item in selection["items"]}
    ):
        raise SystemExit("selection contains a duplicate UID within a comparison group")

    media_dir = args.out / "media"
    frames_dir = args.out / "frames"
    sheets_dir = args.out / "sheets"
    for directory in (media_dir, frames_dir, sheets_dir):
        directory.mkdir(parents=True, exist_ok=True)

    records = []
    for ordinal, selected in enumerate(selection["items"]):
        group = selected.get("group")
        key = (group, selected["query"], selected["uid"])
        source = (
            lookup.get(key)
            if group is not None
            else lookup_without_group.get((selected["query"], selected["uid"]))
        )
        if source is None:
            raise SystemExit(f"selection item absent from enumeration: {key}")
        if source["already_seen"] or (
            source["title_skipped"] and source["uid"] not in allowed_title_skips
        ):
            raise SystemExit(f"selection violates freshness/title contract: {key}")

        uid = source["uid"]
        matches = completed_media(media_dir, uid)
        if not matches:
            download = subprocess.run(
                [
                    args.yt_dlp,
                    "--no-playlist",
                    "--no-progress",
                    "--concurrent-fragments",
                    str(args.concurrent_fragments),
                    "-f",
                    args.format,
                    "--merge-output-format",
                    "mp4",
                    "-o",
                    str(media_dir / f"{uid}.%(ext)s"),
                    source["url"],
                ],
                capture_output=True,
                text=True,
            )
            if download.returncode != 0:
                records.append({
                    "ordinal": ordinal,
                    "pillar": selected.get("pillar"),
                    "selection_reason": selected.get("reason"),
                    "title_skip_audit_override": uid in allowed_title_skips,
                    **source,
                    "artifact_status": "download_failed",
                    "download_error": (download.stderr or download.stdout)[-1000:],
                })
                continue
            matches = completed_media(media_dir, uid)
        if len(matches) != 1:
            raise RuntimeError(f"expected one media artifact for {uid}, got {matches}")
        media = matches[0]
        item_frames_dir = frames_dir / f"{ordinal:02d}_{uid}"
        item_frames_dir.mkdir(parents=True, exist_ok=True)
        try:
            duration, frame_records = extract_frames(
                args.ffmpeg, args.ffprobe, media, item_frames_dir, ordinal, args.max_frames
            )
        except RuntimeError as exc:
            records.append({
                "ordinal": ordinal,
                "pillar": selected.get("pillar"),
                "selection_reason": selected.get("reason"),
                "title_skip_audit_override": uid in allowed_title_skips,
                **source,
                "artifact_status": "render_failed",
                "media_path": str(media.relative_to(args.out)),
                "media_sha256": file_sha256(media),
                "render_error": str(exc)[-1000:],
            })
            continue
        sheet = sheets_dir / f"{ordinal:02d}_{uid}.jpg"
        make_sheet(frame_records, item_frames_dir, sheet)
        records.append({
            "ordinal": ordinal,
            "pillar": selected.get("pillar"),
            "selection_reason": selected.get("reason"),
            "title_skip_audit_override": uid in allowed_title_skips,
            **source,
            "artifact_status": "rendered",
            "media_path": str(media.relative_to(args.out)),
            "media_sha256": file_sha256(media),
            "probed_duration": duration,
            "frames": [
                {**frame, "path": str((item_frames_dir / Path(frame["path"]).name).relative_to(args.out))}
                for frame in frame_records
            ],
            "sheet_path": str(sheet.relative_to(args.out)),
            "sheet_sha256": file_sha256(sheet),
        })

    manifest = {
        "kind": "multi_pillar_search_shadow_video_audit",
        "selection_sha256": hashlib.sha256(args.selection.read_bytes()).hexdigest(),
        "enumerations": [
            {
                "path": str(path),
                "kind": enumeration.get("kind"),
                "content_sha256": enumeration["content_sha256"],
            }
            for path, enumeration in zip(args.enumeration, enumerations)
        ],
        "records": records,
    }
    manifest["records_sha256"] = sha256_json(records)
    (args.out / "video_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
