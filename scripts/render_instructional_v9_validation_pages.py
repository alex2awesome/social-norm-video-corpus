#!/usr/bin/env python3
"""Render metadata-blind contact pages for the frozen V9 validation selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def panel_label(row: dict[str, Any]) -> str:
    return f"#{int(row['audit_index']):04d}  {row['candidate_id']}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blind-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--panel-width", type=int, default=900)
    parser.add_argument("--per-page", type=int, default=4)
    args = parser.parse_args()
    if args.out_manifest.exists():
        raise SystemExit(f"refusing to overwrite: {args.out_manifest}")
    if args.per_page != 4:
        raise SystemExit("the audited layout currently requires --per-page=4")

    rows = load_jsonl(args.blind_manifest)
    if len({row["audit_index"] for row in rows}) != len(rows):
        raise ValueError("duplicate audit_index")
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate candidate_id")
    rows.sort(key=lambda row: int(row["audit_index"]))
    pages_dir = args.out_dir / "blind_pages"
    sheets_dir = args.out_dir / "blind_sheets"
    pages_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=24)
    page_records = []
    for page_index, start in enumerate(range(0, len(rows), 4)):
        batch = rows[start : start + 4]
        panels = []
        page_items = []
        for row in batch:
            source = Path(row["sheet_path"])
            if sha256(source) != row["sheet_sha256"]:
                raise ValueError(
                    f"storyboard hash mismatch: {row['candidate_id']}"
                )
            link = sheets_dir / f"{int(row['audit_index']):04d}.jpg"
            if link.is_symlink() and not link.exists():
                link.unlink()
            if not link.exists():
                os.symlink(source.resolve(), link)
            image = Image.open(source).convert("RGB")
            height = round(image.height * args.panel_width / image.width)
            image = image.resize(
                (args.panel_width, height),
                Image.Resampling.LANCZOS,
            )
            panel = Image.new(
                "RGB",
                (args.panel_width, height + 44),
                (14, 14, 14),
            )
            panel.paste(image, (0, 44))
            ImageDraw.Draw(panel).text(
                (10, 8),
                panel_label(row),
                fill="white",
                font=font,
            )
            panels.append(panel)
            page_items.append(
                {
                    "audit_index": row["audit_index"],
                    "candidate_id": row["candidate_id"],
                }
            )
        while len(panels) < 4:
            panels.append(
                Image.new("RGB", panels[0].size, (14, 14, 14))
            )
        width, height = panels[0].size
        page = Image.new("RGB", (width * 2, height * 2), (8, 8, 8))
        for slot, panel in enumerate(panels):
            page.paste(panel, ((slot % 2) * width, (slot // 2) * height))
        target = pages_dir / f"page_{page_index:03d}.jpg"
        page.save(target, quality=92)
        page_records.append(
            {
                "page_index": page_index,
                "page_path": str(target),
                "page_sha256": sha256(target),
                "items": page_items,
            }
        )
    payload = {
        "kind": "instructional_v9_metadata_blind_review_pages",
        "items": len(rows),
        "pages": page_records,
        "semantic_metadata_emitted": False,
        "source_manifest_sha256": sha256(args.blind_manifest),
    }
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {"items": len(rows), "pages": len(page_records)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
