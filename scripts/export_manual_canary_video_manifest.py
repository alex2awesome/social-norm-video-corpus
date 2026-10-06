#!/usr/bin/env python3
"""Join a frozen manual canary to rendered proxies for video-VLM evaluation."""

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
    vlm_manifest: list[dict[str, Any]],
    manual_review: list[dict[str, Any]],
    proxy_manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    reviews = {row["item_id"]: row for row in manual_review}
    proxies = {
        row["item_id"]: row for row in proxy_manifest.get("records") or []
    }
    if len(reviews) != len(manual_review):
        raise ValueError("manual review contains duplicate item_id values")
    if len(proxies) != len(proxy_manifest.get("records") or []):
        raise ValueError("proxy manifest contains duplicate item_id values")
    item_ids = {row["item_id"] for row in vlm_manifest}
    if set(reviews) != item_ids or set(proxies) != item_ids:
        raise ValueError("manual, proxy, and VLM manifests must cover the same items")
    records = []
    for row in sorted(vlm_manifest, key=lambda value: value["ordinal"]):
        review = reviews[row["item_id"]]
        proxy = proxies[row["item_id"]]
        identity = (int(row["ordinal"]), row["uid"])
        if identity != (int(review["ordinal"]), review["uid"]):
            raise ValueError(f"manual identity mismatch: {row['item_id']}")
        if identity != (int(proxy["ordinal"]), proxy["uid"]):
            raise ValueError(f"proxy identity mismatch: {row['item_id']}")
        social_demo = (
            review["visual_demo_present"] == "yes"
            and review["is_social_norm"] == "yes"
        )
        label_matched = social_demo and review["norm_supported"] == "yes"
        records.append(
            {
                **row,
                "source_clip": proxy["source_clip_path"],
                "proxy_clip": proxy["proxy_path"],
                "proxy_sha256": proxy["proxy_sha256"],
                "gold_scene_visible": review["visual_demo_present"] == "yes",
                "gold_social_scene_visible": social_demo,
                "gold_label_matched_visible": label_matched,
                "gold_usable": label_matched,
                "gold_keep_after_relabel": social_demo,
                "manual_decision": review["decision"],
                "manual_evidence": review["evidence"],
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vlm-manifest", type=Path, required=True)
    parser.add_argument("--manual-review", type=Path, required=True)
    parser.add_argument("--proxy-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = build_records(
        load_jsonl(args.vlm_manifest),
        load_jsonl(args.manual_review),
        json.loads(args.proxy_manifest.read_text()),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"records": len(records)}, sort_keys=True))


if __name__ == "__main__":
    main()
