#!/usr/bin/env python3
"""Export read-only post-reveal evidence for a frozen instructional audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.export_witnessed_semantic_packets import (
        read_jsonl,
        sha256,
        title_map,
        transcript_excerpt,
    )
else:
    from export_witnessed_semantic_packets import (
        read_jsonl,
        sha256,
        title_map,
        transcript_excerpt,
    )


def parse_item_id(item_id: str) -> tuple[str, int]:
    parts = item_id.split(":")
    if len(parts) != 3 or parts[0] != "instructional":
        raise ValueError(f"invalid instructional item_id: {item_id!r}")
    try:
        demo_ordinal = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"invalid demo ordinal in item_id: {item_id!r}") from exc
    if demo_ordinal < 0:
        raise ValueError(f"negative demo ordinal in item_id: {item_id!r}")
    return parts[1], demo_ordinal


def export_packets(root: Path, selection_path: Path) -> list[dict[str, Any]]:
    selection = read_jsonl(selection_path)
    ids = [row.get("item_id") for row in selection]
    if len(ids) != len(set(ids)):
        raise ValueError("selection contains duplicate item_id values")
    parsed = [parse_item_id(str(item_id)) for item_id in ids]
    titles = title_map(root / "data" / "state.db", {uid for uid, _ in parsed})
    packets = []
    for selected, (uid, demo_ordinal) in zip(selection, parsed, strict=True):
        metadata_path = root / "data" / "instructional" / uid / "metadata.json"
        if not metadata_path.is_file():
            raise ValueError(f"{uid}: missing instructional metadata")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        demos = metadata.get("demos")
        if not isinstance(demos, list) or demo_ordinal >= len(demos):
            raise ValueError(f"{selected['item_id']}: demo ordinal is out of range")
        demo = demos[demo_ordinal]
        start = demo.get("start")
        end = demo.get("end")
        if (
            not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or float(start) >= float(end)
        ):
            raise ValueError(f"{selected['item_id']}: invalid demo bounds")
        excerpt, segments = transcript_excerpt(
            root / "data" / "transcripts" / f"{uid}.json",
            max(0.0, float(start) - 3.0),
            float(end) + 3.0,
        )
        provenance = metadata.get("provenance") or {}
        packets.append(
            {
                "item_id": selected["item_id"],
                "audit_index": selected.get("audit_index"),
                "uid": uid,
                "demo_ordinal": demo_ordinal,
                "title_record": titles.get(uid, {}),
                "selection": {
                    "assigned_norm": selected.get("norm"),
                    "polarity": (selected.get("stratum") or {}).get("polarity"),
                    "category": (selected.get("stratum") or {}).get("category"),
                    "found_by_query": selected.get("found_by_query"),
                    "query_source": selected.get("query_source"),
                    "score_band": selected.get("score_band"),
                },
                "provenance": {
                    "category": provenance.get("category"),
                    "found_by_query": provenance.get("found_by_query"),
                    "query_source": provenance.get("query_source"),
                    "genre": metadata.get("genre"),
                },
                "demo": {
                    "start": float(start),
                    "end": float(end),
                    "clip": demo.get("clip"),
                    "polarity": demo.get("polarity"),
                    "norm": demo.get("norm"),
                    "start_quote": demo.get("start_quote"),
                    "end_quote": demo.get("end_quote"),
                    "explanation": demo.get("explanation"),
                },
                "transcript_excerpt": excerpt,
                "transcript_segments": segments,
                "metadata_path": str(metadata_path.relative_to(root)),
            }
        )
    return packets


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    packets = export_packets(args.root, args.selection)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "items": len(packets),
                "selection_sha256": sha256(args.selection),
                "output": str(args.output),
                "policy": "read_only_export",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
