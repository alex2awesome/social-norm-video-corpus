#!/usr/bin/env python3
"""Export a frozen, non-destructive benchmark from the visual-audit ledger.

The benchmark is deliberately source-disjoint: at most one item is selected per
source UID for each pillar/label cell.  It can optionally create small video
proxies for transfer to a model-serving host.  Original clips and metadata are
never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sqlite3
import subprocess
from collections import Counter
from pathlib import Path


def latest_rows(conn: sqlite3.Connection) -> list[dict]:
    conn.row_factory = sqlite3.Row
    instructional = conn.execute(
        """
        SELECT i.*, j.decision, j.is_social_norm, j.visual_demo_present,
               j.medium, j.norm_supported, j.label_leak_visible,
               j.off_topic, j.description
        FROM judgments j
        JOIN items i USING(item_id)
        WHERE j.judgment_id IN (
          SELECT MAX(judgment_id) FROM judgments
          WHERE rubric_version='instructional_v4' GROUP BY item_id
        )
        AND i.has_clip=1 AND i.present=1
        """
    ).fetchall()
    witnessed = conn.execute(
        """
        SELECT i.*, j.decision, j.is_social_norm, j.social_action_visible,
               j.reaction_visible, j.action_before_reaction,
               j.reaction_targets_action, j.reaction_is_normative,
               j.behavior_label_supported, j.clean_pre_reaction_demo,
               j.authenticity, j.description
        FROM witnessed_judgments j
        JOIN items i USING(item_id)
        WHERE j.judgment_id IN (
          SELECT MAX(judgment_id) FROM witnessed_judgments
          WHERE rubric_version='witnessed_v1' GROUP BY item_id
        )
        AND i.has_clip=1 AND i.present=1
        """
    ).fetchall()

    out: list[dict] = []
    for row in instructional:
        item = dict(row)
        value = item["visual_demo_present"]
        if value not in {"yes", "no"}:
            continue
        item["gold_scene_visible"] = value == "yes"
        item["gold_social_scene_visible"] = (
            value == "yes" and item["is_social_norm"] == "yes"
        )
        item["gold_label_matched_visible"] = (
            item["gold_social_scene_visible"] and item["norm_supported"] == "yes"
        )
        item["gold_usable"] = item["decision"] in {"accept", "accept_with_repairs"}
        item["gold_rubric"] = "instructional_v4"
        out.append(item)
    for row in witnessed:
        item = dict(row)
        value = item["social_action_visible"]
        if value not in {"yes", "no"}:
            continue
        item["gold_scene_visible"] = value == "yes"
        item["gold_social_scene_visible"] = (
            value == "yes" and item["is_social_norm"] == "yes"
        )
        item["gold_label_matched_visible"] = (
            item["gold_social_scene_visible"]
            and item["behavior_label_supported"] == "yes"
        )
        item["gold_usable"] = item["decision"] == "accept"
        item["gold_rubric"] = "witnessed_v1"
        out.append(item)
    return out


def balanced_source_disjoint(
    rows: list[dict], per_cell: int, seed: str
) -> list[dict]:
    rng = random.Random(seed)
    cells: dict[tuple[str, bool], list[dict]] = {}
    for row in rows:
        cells.setdefault((row["pillar"], row["gold_scene_visible"]), []).append(row)

    selected: list[dict] = []
    for cell in sorted(cells):
        candidates = cells[cell][:]
        rng.shuffle(candidates)
        seen_uids: set[str] = set()
        for row in candidates:
            if row["uid"] in seen_uids:
                continue
            selected.append(row)
            seen_uids.add(row["uid"])
            if len(seen_uids) >= per_cell:
                break
    return sorted(selected, key=lambda r: r["item_id"])


def proxy_name(item_id: str) -> str:
    digest = hashlib.sha256(item_id.encode()).hexdigest()[:12]
    safe = item_id.replace(":", "__").replace("/", "_")
    return f"{safe[:100]}__{digest}.mp4"


def render_proxy(source: Path, target: Path, ffmpeg: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-an",
        "-vf",
        "fps=3,scale='min(640,iw)':-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "28",
        "-movflags",
        "+faststart",
        str(target),
    ]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-cell", type=int, default=40)
    parser.add_argument("--seed", default="visual-scene-benchmark-v1")
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Exclude all source UIDs present in another benchmark manifest.",
    )
    parser.add_argument("--render-proxies", action="store_true")
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    excluded_uids: set[str] = set()
    for path in args.exclude_manifest:
        if path.exists():
            excluded_uids.update(
                json.loads(line)["uid"]
                for line in path.read_text().splitlines()
                if line.strip()
            )
    rows = balanced_source_disjoint(
        [
            row
            for row in latest_rows(conn)
            if row["uid"] not in excluded_uids
        ],
        per_cell=args.per_cell,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    failures: list[dict] = []
    written: list[dict] = []

    for row in rows:
        record = dict(row)
        source = args.root / record["clip_path"]
        record["source_clip"] = str(source)
        record["source_exists"] = source.is_file()
        proxy = args.out / "clips" / proxy_name(record["item_id"])
        if args.render_proxies and source.is_file():
            try:
                render_proxy(source, proxy, args.ffmpeg)
                record["proxy_clip"] = str(proxy)
            except subprocess.CalledProcessError as exc:
                failures.append({"item_id": record["item_id"], "error": str(exc)})
        elif proxy.is_file():
            record["proxy_clip"] = str(proxy)
        written.append(record)

    with manifest_path.open("w") as handle:
        for record in written:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    summary = {
        "version": "visual_scene_benchmark_v1",
        "seed": args.seed,
        "per_cell": args.per_cell,
        "items": len(written),
        "source_disjoint_uids": len({r["uid"] for r in written}),
        "excluded_source_uids": len(excluded_uids),
        "cells": {
            f"{pillar}:{str(label).lower()}": count
            for (pillar, label), count in sorted(
                Counter(
                    (r["pillar"], r["gold_scene_visible"]) for r in written
                ).items()
            )
        },
        "render_failures": failures,
        "manifest": str(manifest_path),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
