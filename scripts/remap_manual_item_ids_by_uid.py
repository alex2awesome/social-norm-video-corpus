#!/usr/bin/env python3
"""Create an auditable manual-review ID crosswalk using unique source UIDs.

This is for frozen reviews whose sheet manifest used a placeholder statement
index while the exported target manifest retained the original statement
index.  Labels are copied unchanged.  The output records every changed ID and
fails unless the UID mapping is complete, one-to-one, and collision-free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def uid_from_item_id(item_id: str) -> str:
    parts = item_id.split(":", 2)
    if len(parts) != 3 or not parts[1]:
        raise ValueError(f"cannot recover UID from item_id: {item_id!r}")
    return parts[1]


def remap_manual_rows(manual_rows: list[dict], target_rows: list[dict]) -> list[dict]:
    targets_by_uid: dict[str, str] = {}
    for row in target_rows:
        uid = row["uid"]
        if uid in targets_by_uid:
            raise ValueError(f"target manifest contains duplicate UID: {uid}")
        targets_by_uid[uid] = row["item_id"]

    remapped = []
    for row in manual_rows:
        uid = row.get("uid") or uid_from_item_id(row["item_id"])
        if uid not in targets_by_uid:
            raise ValueError(f"manual UID absent from target manifest: {uid}")
        updated = dict(row)
        target_item_id = targets_by_uid[uid]
        if target_item_id != row["item_id"]:
            updated["original_manual_item_id"] = row["item_id"]
            updated["item_id_join_correction"] = "unique_uid_to_target_manifest"
            updated["item_id"] = target_item_id
        remapped.append(updated)

    output_ids = [row["item_id"] for row in remapped]
    if len(set(output_ids)) != len(output_ids):
        raise ValueError("UID remap produced duplicate item IDs")
    if set(output_ids) != set(targets_by_uid.values()):
        missing = sorted(set(targets_by_uid.values()) - set(output_ids))
        extra = sorted(set(output_ids) - set(targets_by_uid.values()))
        raise ValueError(f"manual/target ID sets differ: missing={missing}, extra={extra}")
    return remapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    remapped = remap_manual_rows(load_jsonl(args.manual), load_jsonl(args.targets))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in remapped:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "rows": len(remapped),
                "changed_item_ids": sum(
                    "original_manual_item_id" in row for row in remapped
                ),
                "out": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
