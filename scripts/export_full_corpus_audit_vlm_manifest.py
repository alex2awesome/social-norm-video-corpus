#!/usr/bin/env python3
"""Join frozen human audits to corpus media for an append-only VLM benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_full_corpus_audit_features import load_gold
else:
    from evaluate_full_corpus_audit_features import load_gold


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = str(row["item_id"])
        if item_id in result:
            raise ValueError(f"duplicate item_id: {item_id}")
        result[item_id] = row
    return result


def clip_relative_transcript(packet: dict[str, Any]) -> list[dict[str, Any]]:
    segments = packet.get("transcript_segments") or []
    clip_window = packet.get("clip_window")
    offset = (
        float(clip_window[0])
        if isinstance(clip_window, list) and len(clip_window) == 2
        else 0.0
    )
    result = []
    for segment in segments:
        try:
            result.append(
                {
                    "start": max(0.0, float(segment["start"]) - offset),
                    "end": max(0.0, float(segment["end"]) - offset),
                    "text": str(segment["text"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return result


def export(
    audit_root: Path,
    corpus_manifests: dict[str, Path],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    ordinal = 0
    for cohort in ("mechanism", "uniform"):
        for pillar in ("instructional", "witnessed", "commentary"):
            root = audit_root / cohort / pillar
            selection = read_jsonl(root / "sealed_selection.jsonl")
            packets = keyed(read_jsonl(root / "post_reveal_semantic_packets.jsonl"))
            corpus = keyed(read_jsonl(corpus_manifests[pillar]))
            gold = load_gold(audit_root, cohort, pillar)
            for selected in sorted(
                selection, key=lambda row: int(row["audit_index"])
            ):
                item_id = str(selected["item_id"])
                source = corpus[item_id]
                packet = packets[item_id]
                record = {
                    "ordinal": ordinal,
                    "cohort": cohort,
                    "audit_index": int(selected["audit_index"]),
                    "item_id": item_id,
                    "pillar": pillar,
                    "uid": source["uid"],
                    "source_clip": source["source_clip"],
                    "duration_hint": source.get("duration_hint"),
                    "media_start_sec": source.get("media_start_sec"),
                    "media_end_sec": source.get("media_end_sec"),
                    "norm": source.get("norm"),
                    "explanation": source.get("explanation"),
                    "gold_scene_visible": bool(gold[item_id]["visual"]),
                    "gold_social_scene_visible": bool(gold[item_id]["visual"]),
                    "gold_label_matched_visible": bool(gold[item_id]["strict"]),
                    "gold_usable": bool(gold[item_id]["strict"]),
                    "aligned_transcript": clip_relative_transcript(packet),
                }
                if not Path(str(record["source_clip"])).is_file():
                    raise FileNotFoundError(record["source_clip"])
                records.append(record)
                ordinal += 1
    if len(records) != 300 or len({row["uid"] for row in records}) != 300:
        raise ValueError(
            f"expected 300 source-disjoint records, got "
            f"{len(records)} records/{len({row['uid'] for row in records})} UIDs"
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--instructional-manifest", required=True, type=Path)
    parser.add_argument("--witnessed-manifest", required=True, type=Path)
    parser.add_argument("--commentary-manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    rows = export(
        args.audit_root,
        {
            "instructional": args.instructional_manifest,
            "witnessed": args.witnessed_manifest,
            "commentary": args.commentary_manifest,
        },
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "items": len(rows),
                "source_disjoint_uids": len({row["uid"] for row in rows}),
                "policy": "frozen_read_only_benchmark_export",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
