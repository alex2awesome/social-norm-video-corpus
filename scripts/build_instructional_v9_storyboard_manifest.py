#!/usr/bin/env python3
"""Join a frozen validation selection to its label-free dense storyboard paths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(
    selection_path: Path,
    dense_manifest_path: Path,
    sheet_root: Path,
) -> list[dict]:
    selection = {
        int(row["audit_index"]): row for row in load_jsonl(selection_path)
    }
    dense = {
        int(row["audit_index"]): row for row in load_jsonl(dense_manifest_path)
    }
    if set(selection) != set(dense):
        raise ValueError("selection and dense manifests have different audit indices")
    rows = []
    for index in sorted(selection):
        selected = selection[index]
        storyboard = dense[index]
        if selected["uid"] != storyboard["uid"]:
            raise ValueError(f"audit_index {index}: UID mismatch")
        sheet_name = Path(storyboard["sheet_path"]).name
        rows.append(
            {
                "audit_index": index,
                "item_id": selected["item_id"],
                "uid": selected["uid"],
                "pillar": selected["pillar"],
                "sheet_path": str(sheet_root / sheet_name),
                "sheet_sha256": storyboard["sheet_sha256"],
                "frame_count": storyboard["media"]["frame_count"],
                "sampling_fps": storyboard["media"]["fps"],
                "frame_order": "left_to_right_then_top_to_bottom",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dense-manifest", type=Path, required=True)
    parser.add_argument("--sheet-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.selection, args.dense_manifest, args.sheet_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
