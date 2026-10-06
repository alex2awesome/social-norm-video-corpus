#!/usr/bin/env python3
"""Cut exact pre-reaction action clips from audited window proposals.

Consumes witnessed_action_window_proposer_v1 proposals plus the LF record
stream, selects the organic tier (no staging / authority / WWYD / visual
staged-role-play negative evidence), and re-encodes exact
``[action_start, action_end)`` clips from the existing saved clip files into a
separate derived directory.  Source media is never modified or deleted; every
cut is manifested with parameters and SHA-256.  The 2026-08-22 16-item frame
audit found the window correct on every organic item and all failures to be
source-type contamination, which this tier filter targets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

CUTTER_VERSION = "cut_witnessed_action_clips_v1"

# Negative provenance evidence that removes an item from the organic tier.
ORGANIC_EXCLUSION_LFS = {
    "registry_witnessed_creator_staging_title_v2",
    "registry_witnessed_official_wwyd_channel_v1",
    "registry_witnessed_authority_exact_span_v3",
    "vis_witnessed_staged_roleplay_v1",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def excluded_items(lf_records: Iterable[dict[str, Any]]) -> set[str]:
    excluded = set()
    for record in lf_records:
        if record.get("lf_id") in ORGANIC_EXCLUSION_LFS and record.get("vote") == -1:
            excluded.add(record["item_id"])
    return excluded


def select_proposals(
    proposals: Iterable[dict[str, Any]],
    excluded: set[str],
    *,
    tier: str,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected, counts = [], {"proposed": 0, "excluded_provenance": 0, "not_proposed": 0}
    for row in proposals:
        if row.get("status") != "proposed":
            counts["not_proposed"] += 1
            continue
        counts["proposed"] += 1
        if tier == "organic" and row["item_id"] in excluded:
            counts["excluded_provenance"] += 1
            continue
        selected.append(row)
    return selected, counts


def cut_clip(
    ffmpeg: str, source: Path, dest: Path, start: float, end: float
) -> tuple[bool, str | None]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-ss", f"{max(0.0, start):.3f}", "-i", str(source),
        "-t", f"{max(0.0, end - start):.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-movflags", "+faststart", str(tmp),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False, (result.stderr or "empty output")[-300:]
    tmp.replace(dest)
    return True, None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--lf-records", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--tier", choices=("organic", "all"), default="organic")
    parser.add_argument("--view", choices=("action", "context"), default="action",
                        help="action: tight pre-reaction window from the saved clip. "
                             "context: --context-pre-sec before the reaction anchor, "
                             "cut from the retained RAW source (reaction still excluded)")
    parser.add_argument("--context-pre-sec", type=float, default=30.0)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    excluded = excluded_items(iter_jsonl(args.lf_records))
    selected, counts = select_proposals(
        iter_jsonl(args.proposals), excluded, tier=args.tier
    )
    suffix = "" if args.view == "action" else "_context"
    manifest_path = args.out_dir / f"cut_manifest{suffix}_shard_{args.shard_index}.jsonl"
    done = set()
    if manifest_path.exists():
        done = {row["item_id"] for row in iter_jsonl(manifest_path) if row.get("ok")}

    cut = failed = skipped = 0
    with manifest_path.open("a") as manifest:
        for row in selected:
            digest = int(hashlib.sha256(row["item_id"].encode()).hexdigest()[:8], 16)
            if digest % args.num_shards != args.shard_index:
                continue
            if row["item_id"] in done:
                skipped += 1
                continue
            if args.limit is not None and cut >= args.limit:
                break
            uid, clip_idx = row["uid"], row["clip_idx"]
            if args.view == "context":
                # Longer lead-in cut from the retained raw source, in source
                # coordinates; still ends before the reaction anchor.
                anchor = row["reaction_anchor_sec"]
                rel_start = max(0.0, anchor - args.context_pre_sec)
                rel_end = row["action_window_sec"][1]
                candidates = sorted(
                    (args.root / "data" / "raw_video").glob(f"{uid}.*")
                )
                source = next(
                    (p for p in candidates if p.suffix.lower() in
                     {".mp4", ".mkv", ".webm", ".mov", ".m4v"}),
                    args.root / "data" / "raw_video" / f"{uid}.mp4",
                )
                dest = (args.out_dir / "clips_context" / uid /
                        f"clip_{clip_idx}_context.mp4")
            else:
                source = args.root / "data" / "hits" / uid / f"clip_{clip_idx}.mp4"
                dest = args.out_dir / "clips" / uid / f"clip_{clip_idx}_action.mp4"
                rel_start, rel_end = row["action_window_clip_relative_sec"]
            if not source.is_file():
                ok, error = False, "source clip missing"
            else:
                ok, error = cut_clip(args.ffmpeg, source, dest, rel_start, rel_end)
            manifest.write(json.dumps({
                "item_id": row["item_id"], "uid": uid, "clip_idx": clip_idx,
                "ok": ok, "error": error,
                "source": str(source), "dest": str(dest) if ok else None,
                "sha256": sha256_file(dest) if ok else None,
                "action_window_clip_relative_sec": row["action_window_clip_relative_sec"],
                "action_window_sec": row["action_window_sec"],
                "cut_window_sec": [rel_start, rel_end],
                "view": args.view,
                "audio_snap": row.get("audio_snap"),
                "cutter_version": CUTTER_VERSION, "tier": args.tier,
                "source_preserved": True,
            }, sort_keys=True) + "\n")
            cut += ok
            failed += not ok
    summary = {
        "cutter_version": CUTTER_VERSION, "tier": args.tier,
        "shard": [args.shard_index, args.num_shards],
        **counts, "cut": cut, "failed": failed, "already_done": skipped,
        "source_preserved": True, "deleted": 0,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
