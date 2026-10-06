#!/usr/bin/env python3
"""Resolve sealed transfer pointers into label-blind transcript packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.select_commentary_occurred_event_transfer_v3 import read_jsonl, sha256
else:
    from select_commentary_occurred_event_transfer_v3 import read_jsonl, sha256


FORBIDDEN = {
    "uid",
    "item_id",
    "title",
    "category",
    "query_source",
    "found_by_query",
    "signal",
    "polarity",
    "norm",
    "v1_decision",
    "v1_normalized_behavior",
    "v1_normalized_norm",
}


def build_packets(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifests: dict[str, dict[str, dict[str, Any]]] = {}
    packets = []
    for row in selected:
        path = Path(row["source_manifest_path"])
        if sha256(path) != row.get("source_manifest_sha256"):
            raise ValueError(f"transfer index {row.get('transfer_index')}: source hash changed")
        key = str(path)
        if key not in manifests:
            manifest = json.loads(path.read_text())
            manifests[key] = {
                item["item_id"]: item for item in manifest.get("items") or []
            }
        item = manifests[key].get(row["item_id"])
        if item is None:
            raise ValueError(f"{row['item_id']}: missing linked manifest item")
        context = item.get("transcript_context") or []
        if not context or not all(str(segment.get("text") or "").strip() for segment in context):
            raise ValueError(f"{row['item_id']}: empty transcript context")
        packet = {
            "blind_id": f"commentary-transfer-v3-{int(row['transfer_index']):04d}",
            "transfer_index": int(row["transfer_index"]),
            "content_manifest_sha256": row.get("content_manifest_sha256") or "",
            "detector_start_sec": float(item.get("start_sec") or 0),
            "detector_end_sec": float(item.get("end_sec") or 0),
            "transcript_context": [
                {
                    "start": float(segment.get("start") or 0),
                    "end": float(segment.get("end") or 0),
                    "text": str(segment.get("text") or ""),
                }
                for segment in context
            ],
        }
        if set(packet) & FORBIDDEN:
            raise AssertionError("blind packet leaks forbidden retrieval metadata")
        packets.append(packet)
    if [row["transfer_index"] for row in packets] != list(range(len(packets))):
        raise ValueError("blind packet indices are incomplete or reordered")
    return packets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    args = parser.parse_args()
    for row in build_packets(read_jsonl(args.selection)):
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
