#!/usr/bin/env python3
"""Render compact labeled sheets for a frame-only hourly visual audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def shorten(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_page(
    batch: Path,
    rows: list[dict],
    target: Path,
    *,
    cell_width: int = 320,
    image_height: int = 180,
    label_height: int = 76,
) -> None:
    columns = 3
    sheet = Image.new(
        "RGB",
        (columns * cell_width, len(rows) * (image_height + label_height)),
        (18, 18, 18),
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for row_index, row in enumerate(rows):
        y0 = row_index * (image_height + label_height)
        frame_names = row.get("frames") or []
        for column, frame_name in enumerate(frame_names[:columns]):
            image = Image.open(batch / "frames" / frame_name).convert("RGB")
            image.thumbnail((cell_width, image_height))
            x = column * cell_width + (cell_width - image.width) // 2
            y = y0 + (image_height - image.height) // 2
            sheet.paste(image, (x, y))
        label_y = y0 + image_height + 2
        draw.text(
            (4, label_y),
            f"{int(row['audit_index']):03d} {row['pillar']} | "
            f"{shorten(row.get('title'), 90)}",
            fill="white",
            font=font,
        )
        draw.text(
            (4, label_y + 18),
            f"norm: {shorten(row.get('norm'), 120)}",
            fill=(220, 220, 120),
            font=font,
        )
        draw.text(
            (4, label_y + 36),
            f"statement/reaction: "
            f"{shorten(row.get('statement') or row.get('reaction'), 130)}",
            fill=(180, 220, 255),
            font=font,
        )
        draw.text(
            (4, label_y + 54),
            f"query: {shorten(row.get('query'), 130)}",
            fill=(190, 190, 190),
            font=font,
        )
    sheet.save(target, quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--items-per-page", type=int, default=8)
    args = parser.parse_args()
    if args.items_per_page <= 0:
        raise SystemExit("items-per-page must be positive")
    batch = args.batch.resolve()
    manifest = json.loads((batch / "manifest.json").read_text())
    rows = manifest.get("visual_samples") or []
    out = batch / "sheets"
    out.mkdir(exist_ok=True)
    pages = []
    for start in range(0, len(rows), args.items_per_page):
        target = out / f"page_{start // args.items_per_page:02d}.jpg"
        render_page(batch, rows[start : start + args.items_per_page], target)
        pages.append(str(target.relative_to(batch)))
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "source_manifest": "manifest.json",
                "items": len(rows),
                "items_per_page": args.items_per_page,
                "pages": pages,
            },
            indent=2,
        )
        + "\n"
    )
    print(json.dumps({"items": len(rows), "pages": len(pages)}))


if __name__ == "__main__":
    main()
