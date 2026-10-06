#!/usr/bin/env python3
"""Re-tile tall metadata-blind dense sheets for VLM and manual inspection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_panels(
    image: np.ndarray,
    panel_count: int,
    columns: int,
) -> np.ndarray:
    if panel_count < 1 or columns < 1 or panel_count % columns:
        raise ValueError("panel_count must be positive and divisible by columns")
    height, width = image.shape[:2]
    if height % panel_count:
        raise ValueError("sheet height is not divisible by panel count")
    panel_height = height // panel_count
    panels = [
        image[index * panel_height : (index + 1) * panel_height]
        for index in range(panel_count)
    ]
    rows = []
    for start in range(0, panel_count, columns):
        rows.append(np.hstack(panels[start : start + columns]))
    compact = np.vstack(rows)
    expected = (panel_height * (panel_count // columns), width * columns)
    if compact.shape[:2] != expected:
        raise RuntimeError("unexpected compact sheet dimensions")
    return compact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--panel-count", type=int, default=8)
    parser.add_argument("--columns", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=84)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    args.out.mkdir(parents=True)
    sheets = args.out / "sheets"
    sheets.mkdir()

    output = []
    failures = []
    for row in load_jsonl(args.manifest):
        if row.get("error") or not row.get("sheet_path"):
            failures.append(row)
            continue
        source = Path(row["sheet_path"])
        if not source.is_absolute():
            source = (
                source
                if source.exists()
                else args.manifest.parent / source
            )
        if sha256(source) != row["sheet_sha256"]:
            raise ValueError(f"source hash mismatch: {row['candidate_id']}")
        image = cv2.imread(str(source))
        if image is None:
            raise ValueError(f"unreadable sheet: {source}")
        compact = compact_panels(image, args.panel_count, args.columns)
        target = sheets / f"{row['candidate_id']}.jpg"
        if not cv2.imwrite(
            str(target),
            compact,
            [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
        ):
            raise RuntimeError(f"failed to write: {target}")
        output.append(
            {
                **row,
                "item_id": row["candidate_id"],
                "sheet_path": str(target),
                "sheet_sha256": sha256(target),
                "source_sheet_path": str(source),
                "source_sheet_sha256": row["sheet_sha256"],
                "panel_count": args.panel_count,
                "panel_columns": args.columns,
            }
        )

    manifest = args.out / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output)
    )
    failures_path = args.out / "render_failures.jsonl"
    failures_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in failures)
    )
    summary = {
        "kind": "compact_blind_dense_sheets",
        "items": len(output),
        "failed_input_items": len(failures),
        "panel_count": args.panel_count,
        "panel_columns": args.columns,
        "jpeg_quality": args.jpeg_quality,
        "manifest_sha256": sha256(manifest),
        "semantic_metadata_present": False,
        "corpus_mutated": False,
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
