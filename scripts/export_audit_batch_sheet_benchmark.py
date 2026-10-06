#!/usr/bin/env python3
"""Adapt a frozen visual-audit batch to a hash-checked sheet VLM manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_records(batch: Path) -> list[dict]:
    source = json.loads((batch / "manifest.json").read_text())
    sheets = json.loads((batch / "sheets" / "manifest.json").read_text())
    source_items = {row["item_id"]: row for row in source["items"]}
    if len(source_items) != len(source["items"]):
        raise ValueError("source manifest has duplicate item_id values")

    output = []
    for sheet_item in sheets["items"]:
        item_id = sheet_item["item_id"]
        if item_id not in source_items:
            raise ValueError(f"sheet item missing from source manifest: {item_id}")
        source_item = source_items[item_id]
        if sheet_item["frame_manifest_sha256"] != source_item["frame_manifest_sha256"]:
            raise ValueError(f"frame manifest mismatch: {item_id}")
        if len(sheet_item["pages"]) != 1:
            raise ValueError(f"expected one contact-sheet page: {item_id}")
        page = sheet_item["pages"][0]
        sheet = batch / page["path"]
        if not sheet.is_file() or file_sha256(sheet) != page["sha256"]:
            raise ValueError(f"sheet hash mismatch: {item_id}")
        norm = str(source_item.get("norm") or "").strip()
        explanation = str(source_item.get("explanation") or "").strip()
        hypothesis = norm
        if explanation:
            hypothesis = f"{norm}; detector explanation: {explanation[:800]}"
        output.append(
            {
                "item_id": item_id,
                "ordinal": source_item["ordinal"],
                "uid": source_item["uid"],
                "pillar": source_item["pillar"],
                "category": source_item.get("category"),
                "polarity": source_item.get("polarity"),
                "found_by_query": source_item.get("found_by_query"),
                "norm": norm,
                "explanation": explanation,
                "query": hypothesis,
                "sheet_path": str(sheet.resolve()),
                "sheet_sha256": page["sha256"],
                "frame_manifest_sha256": source_item["frame_manifest_sha256"],
                "gold_visual_content_present": None,
                "gold_social_behavior_scene": None,
                "gold_target_source_pass": None,
            }
        )
    return sorted(output, key=lambda row: row["ordinal"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    records = build_records(args.batch.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    print(json.dumps({"items": len(records), "out": str(args.out)}, sort_keys=True))


if __name__ == "__main__":
    main()
