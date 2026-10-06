#!/usr/bin/env python3
"""Render label-blind temporal sheets and review pages from an audit manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def render_item(batch: Path, row: dict[str, Any], target: Path) -> None:
    frame_names = row.get("frames") or []
    if not frame_names:
        raise ValueError(f"item {row.get('audit_index')} has no frames")
    columns = 4
    cell_width, cell_height = 400, 250
    header_height = 34
    rows = (len(frame_names) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * cell_width, header_height + rows * cell_height),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (8, 9),
        f"item {int(row['audit_index']):03d} | {row['uid']} | "
        f"{len(frame_names)} ordered frames",
        fill="white",
        font=font,
    )
    for index, frame_name in enumerate(frame_names):
        image = Image.open(batch / "frames" / frame_name).convert("RGB")
        image.thumbnail((cell_width, cell_height - 22))
        x0 = (index % columns) * cell_width
        y0 = header_height + (index // columns) * cell_height
        x = x0 + (cell_width - image.width) // 2
        y = y0 + 20 + (cell_height - 20 - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text((x0 + 6, y0 + 4), f"frame {index:02d}", fill="white", font=font)
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, quality=91)


def render_page(item_sheets: list[Path], target: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in item_sheets]
    width = max(image.width for image in images)
    height = sum(image.height for image in images)
    page = Image.new("RGB", (width, height), (8, 8, 8))
    y = 0
    for image in images:
        page.paste(image, (0, y))
        y += image.height
    target.parent.mkdir(parents=True, exist_ok=True)
    page.save(target, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--items-per-page", type=int, default=3)
    args = parser.parse_args()
    if args.items_per_page <= 0:
        raise SystemExit("items-per-page must be positive")
    batch = args.batch.resolve()
    manifest = json.loads((batch / "manifest.json").read_text())
    rows = manifest.get("visual_samples") or manifest.get("items") or []
    item_dir = batch / "blind_item_sheets"
    page_dir = batch / "blind_review_pages"
    if item_dir.exists() or page_dir.exists():
        raise SystemExit("blind sheet output already exists")

    item_records = []
    for row in rows:
        ordinal = int(row.get("audit_index", row.get("ordinal")))
        target = item_dir / f"{ordinal:03d}_{row['uid']}.jpg"
        render_item(batch, row, target)
        item_records.append(
            {
                "ordinal": ordinal,
                "uid": row["uid"],
                "path": str(target.relative_to(batch)),
                "sha256": file_sha256(target),
            }
        )

    pages = []
    for start in range(0, len(item_records), args.items_per_page):
        members = item_records[start : start + args.items_per_page]
        target = page_dir / f"page_{start // args.items_per_page:02d}.jpg"
        render_page([batch / member["path"] for member in members], target)
        pages.append(
            {
                "path": str(target.relative_to(batch)),
                "ordinals": [member["ordinal"] for member in members],
                "sha256": file_sha256(target),
            }
        )
    output = {
        "schema_version": 1,
        "kind": "label_blind_temporal_review_pages",
        "source_manifest": "manifest.json",
        "items": item_records,
        "items_per_page": args.items_per_page,
        "pages": pages,
    }
    (page_dir / "manifest.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"items": len(item_records), "pages": len(pages)}))


if __name__ == "__main__":
    main()
