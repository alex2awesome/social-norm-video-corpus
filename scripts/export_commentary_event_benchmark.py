#!/usr/bin/env python3
"""Render a frozen commentary target-window benchmark from manual reviews."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rendered-manifest", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    args = parser.parse_args()

    records = {
        row["uid"]: row
        for row in json.loads(args.rendered_manifest.read_text())["records"]
    }
    reviews = [
        json.loads(line)
        for line in args.manual_review.read_text().splitlines()
        if line.strip()
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    clips = args.out / "clips"
    clips.mkdir(exist_ok=True)
    output = []
    failures = []
    for review in reviews:
        row = records[review["uid"]]
        start, end = row["target_window"]
        sources = [
            path
            for suffix in ("mp4", "webm", "mkv")
            if (path := args.video_root / f"{row['uid']}.{suffix}").exists()
        ]
        if not sources:
            failures.append({"uid": row["uid"], "error": "source missing"})
            continue
        target = clips / f"{review['ordinal']:02d}__{row['uid']}.mp4"
        try:
            subprocess.run(
                [
                    args.ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{start:.3f}",
                    "-i",
                    str(sources[0]),
                    "-t",
                    f"{end - start:.3f}",
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
                    "-y",
                    str(target),
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            failures.append({"uid": row["uid"], "error": str(exc)})
            continue
        clear = review.get("demo_quality") in {"clear_visual", "clear_audiovisual"}
        on_camera = review.get("behavior_occurrence") in {
            "on_camera_visual_action",
            "on_camera_speech_act",
        }
        grounded = review.get("event_identity_grounded") == "yes"
        positive = clear and on_camera and grounded
        output.append(
            {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": "commentary",
                "title": row.get("title"),
                "category": row.get("category"),
                "norm": row.get("normalized_norm"),
                "normalized_behavior": row.get("normalized_behavior"),
                "explanation": row.get("behavior_evidence_quote"),
                "proxy_clip": str(target.resolve()),
                "target_window": row["target_window"],
                "gold_scene_visible": positive,
                "gold_social_scene_visible": positive,
                "gold_label_matched_visible": positive
                and review.get("text_label_decision") != "reject",
                "gold_usable": review.get("expected_disposition")
                in {"recover_commentary_visual", "reroute_instructional_review"},
                "manual_review": review,
            }
        )
    with (args.out / "manifest.jsonl").open("w") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "items": len(output),
        "positives": sum(row["gold_scene_visible"] for row in output),
        "negatives": sum(not row["gold_scene_visible"] for row in output),
        "failures": failures,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
