#!/usr/bin/env python3
"""Join frozen witnessed manual judgments to compact full-video proxies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def build_records(
    source_manifest: dict[str, Any],
    manual_rows: list[dict[str, Any]],
    proxy_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    items = {row["item_id"]: row for row in source_manifest.get("items") or []}
    manual = {row["item_id"]: row for row in manual_rows}
    proxies = {
        row["item_id"]: row for row in proxy_manifest.get("records") or []
    }
    expected = set(items)
    if (
        len(items) != len(source_manifest.get("items") or [])
        or len(manual) != len(manual_rows)
        or len(proxies) != len(proxy_manifest.get("records") or [])
    ):
        raise ValueError("source, manual, and proxy inputs require unique item IDs")
    if set(manual) != expected or set(proxies) != expected:
        raise ValueError("source, manual, and proxy inputs must cover the same items")

    records = []
    for item in sorted(items.values(), key=lambda row: int(row["ordinal"])):
        item_id = item["item_id"]
        judgment = manual[item_id]
        proxy = proxies[item_id]
        identity = (int(item["ordinal"]), item["uid"])
        if identity != (int(proxy["ordinal"]), proxy["uid"]):
            raise ValueError(f"proxy identity mismatch: {item_id}")
        if judgment["uid"] != item["uid"]:
            raise ValueError(f"manual identity mismatch: {item_id}")
        disposition = judgment["disposition"]
        if disposition not in {
            "strict_accept",
            "recover_strict",
            "reroute_instructional_review",
            "reject",
            "uncertain",
        }:
            raise ValueError(f"unsupported manual disposition: {disposition}")
        records.append(
            {
                "item_id": item_id,
                "ordinal": int(item["ordinal"]),
                "uid": item["uid"],
                "pillar": "witnessed",
                "category": item.get("category"),
                "norm": item.get("norm"),
                "explanation": item.get("explanation"),
                "aligned_transcript": item.get("aligned_transcript") or [],
                "source_clip": proxy["source_clip_path"],
                "proxy_clip": proxy["proxy_path"],
                "proxy_sha256": proxy["proxy_sha256"],
                "manual_disposition": disposition,
                "manual_authenticity": judgment["authenticity"],
                "manual_description": judgment["description"],
                "gold_strict_current": disposition == "strict_accept",
                "gold_witnessed_recovery": disposition
                in {"strict_accept", "recover_strict"},
                "gold_instructional_reroute": disposition
                == "reroute_instructional_review",
                "gold_any_visual_recovery": disposition
                in {
                    "strict_accept",
                    "recover_strict",
                    "reroute_instructional_review",
                },
                # Retain the generic field consumed by the common VLM runner.
                "gold_usable": disposition == "strict_accept",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--proxy-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = build_records(
        json.loads(args.source_manifest.read_text()),
        load_jsonl(args.manual),
        json.loads(args.proxy_manifest.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records), "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
