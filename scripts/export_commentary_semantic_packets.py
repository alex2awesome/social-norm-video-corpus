#!/usr/bin/env python3
"""Export read-only post-reveal evidence for a frozen commentary audit."""

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
    if len(parts) != 3 or parts[0] != "commentary":
        raise ValueError(f"invalid commentary item_id: {item_id!r}")
    try:
        statement_ordinal = int(parts[2])
    except ValueError as exc:
        raise ValueError(f"invalid statement ordinal in item_id: {item_id!r}") from exc
    if statement_ordinal < 0:
        raise ValueError(f"negative statement ordinal in item_id: {item_id!r}")
    return parts[1], statement_ordinal


def unique_rows(rows: list[dict[str, Any]], source: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        item_id = row.get("item_id")
        if not isinstance(item_id, str) or not item_id:
            raise ValueError(f"{source} contains a missing item_id")
        if item_id in result:
            raise ValueError(f"{source} contains duplicate item_id {item_id}")
        result[item_id] = row
    return result


def media_window(blind_row: dict[str, Any]) -> tuple[float, float]:
    media = blind_row.get("media")
    if not isinstance(media, dict):
        raise ValueError(f"{blind_row.get('item_id')}: missing blind media record")
    start = media.get("sample_start_sec")
    end = media.get("sample_end_sec")
    if (
        not isinstance(start, (int, float))
        or not isinstance(end, (int, float))
        or float(start) >= float(end)
    ):
        raise ValueError(f"{blind_row.get('item_id')}: invalid blind media window")
    return float(start), float(end)


def export_packets(
    root: Path,
    selection_path: Path,
    blind_manifest_path: Path,
) -> list[dict[str, Any]]:
    selection = read_jsonl(selection_path)
    blind_rows = read_jsonl(blind_manifest_path)
    selected_by_id = unique_rows(selection, "selection")
    blind_by_id = unique_rows(blind_rows, "blind manifest")
    if set(selected_by_id) != set(blind_by_id):
        missing = sorted(set(selected_by_id) - set(blind_by_id))
        extra = sorted(set(blind_by_id) - set(selected_by_id))
        raise ValueError(
            f"selection/blind coverage mismatch: missing={missing}, extra={extra}"
        )

    parsed = {
        item_id: parse_item_id(item_id)
        for item_id in selected_by_id
    }
    titles = title_map(
        root / "data" / "state.db",
        {uid for uid, _ in parsed.values()},
    )
    packets: list[dict[str, Any]] = []
    for selected in selection:
        item_id = selected["item_id"]
        blind = blind_by_id[item_id]
        uid, statement_ordinal = parsed[item_id]
        if blind.get("uid") != uid:
            raise ValueError(f"{item_id}: blind uid does not match item_id")
        metadata_path = root / "data" / "discussion" / f"{uid}.json"
        if not metadata_path.is_file():
            raise ValueError(f"{uid}: missing commentary metadata")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        statements = metadata.get("statements")
        if (
            not isinstance(statements, list)
            or statement_ordinal >= len(statements)
        ):
            raise ValueError(f"{item_id}: statement ordinal is out of range")
        statement = statements[statement_ordinal]
        statement_start = statement.get("start")
        statement_end = statement.get("end")
        if (
            not isinstance(statement_start, (int, float))
            or not isinstance(statement_end, (int, float))
            or float(statement_start) >= float(statement_end)
        ):
            raise ValueError(f"{item_id}: invalid statement bounds")
        window_start, window_end = media_window(blind)
        excerpt, segments = transcript_excerpt(
            root / "data" / "transcripts" / f"{uid}.json",
            max(0.0, min(window_start, float(statement_start)) - 3.0),
            max(window_end, float(statement_end)) + 3.0,
        )
        provenance = metadata.get("provenance") or {}
        packets.append(
            {
                "item_id": item_id,
                "audit_index": selected.get("audit_index"),
                "uid": uid,
                "statement_ordinal": statement_ordinal,
                "title_record": titles.get(uid, {}),
                "selection": {
                    "assigned_norm": selected.get("norm"),
                    "polarity": (selected.get("stratum") or {}).get("polarity"),
                    "category": (selected.get("stratum") or {}).get("category"),
                    "found_by_query": selected.get("found_by_query"),
                    "query_source": selected.get("query_source"),
                    "score_band": selected.get("score_band"),
                },
                "discussion": {
                    "title": metadata.get("title"),
                    "category": metadata.get("category"),
                    "source": metadata.get("source"),
                    "modality": metadata.get("modality"),
                    "found_by_query": provenance.get("found_by_query"),
                    "query_source": provenance.get("query_source"),
                    "scene": provenance.get("scene"),
                },
                "statement": {
                    "quote": statement.get("quote"),
                    "norm": statement.get("norm"),
                    "signal": statement.get("signal"),
                    "start": float(statement_start),
                    "end": float(statement_end),
                },
                "visual_window": {
                    "start": window_start,
                    "end": window_end,
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
    parser.add_argument("--blind-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    packets = export_packets(args.root, args.selection, args.blind_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "items": len(packets),
                "selection_sha256": sha256(args.selection),
                "blind_manifest_sha256": sha256(args.blind_manifest),
                "output": str(args.output),
                "policy": "read_only_export",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
