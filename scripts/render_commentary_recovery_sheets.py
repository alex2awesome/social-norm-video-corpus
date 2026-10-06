#!/usr/bin/env python3
"""Render overview and quote-window sheets from a commentary recovery audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def render_sheet(root: Path, record: dict, kind: str, columns: int, target: Path) -> None:
    frames = record[f"{kind}_frames"]
    cell_width, cell_height, header = 320, 200, 70
    rows = (len(frames) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell_width, header + rows * cell_height), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text(
        (8, 6),
        "{:02d} {} | {} | {}".format(
            int(record["ordinal"]), record["uid"], kind, record.get("title") or ""
        ),
        fill="white",
        font=font,
    )
    draw.text(
        (8, 26), f"behavior: {record.get('normalized_behavior') or ''}",
        fill="white", font=font,
    )
    draw.text(
        (8, 46), f"norm: {record.get('normalized_norm') or ''}",
        fill="white", font=font,
    )
    for index, frame in enumerate(frames):
        image = Image.open(root / frame["path"]).convert("RGB")
        image.thumbnail((cell_width, 180))
        column, row = index % columns, index // columns
        x = column * cell_width + (cell_width - image.width) // 2
        y = header + row * cell_height + (180 - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text(
            (column * cell_width + 4, header + row * cell_height + 182),
            f"{float(frame['timestamp']):.3f}s",
            fill="white",
            font=font,
        )
    sheet.save(target, quality=88)


def combine_sheets(overview: Path, target: Path, combined: Path) -> None:
    """Place overview and transcript-centered evidence in one review image."""
    left = Image.open(overview).convert("RGB")
    right = Image.open(target).convert("RGB")
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (left.width + right.width, height), (12, 12, 12))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    canvas.save(combined, quality=88)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    root = args.batch.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    out = root / "sheets"
    out.mkdir(exist_ok=True)
    for record in manifest["records"]:
        prefix = f"{int(record['ordinal']):02d}_{record['uid']}"
        overview = out / f"{prefix}_overview.jpg"
        target = out / f"{prefix}_target.jpg"
        render_sheet(root, record, "overview", 4, overview)
        render_sheet(root, record, "target", 6, target)
        combine_sheets(overview, target, out / f"{prefix}_combined.jpg")
    print(
        json.dumps(
            {
                "records": len(manifest["records"]),
                "sheets": len(list(out.glob("*.jpg"))),
            }
        )
    )


if __name__ == "__main__":
    main()
