#!/usr/bin/env python3
"""Collate per-item shadow-band sheets into legible manual-review pages."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sheets-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--items-per-page", type=int, default=4)
    args = parser.parse_args()
    payload = json.loads(args.sheets_manifest.read_text())
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in payload.get("items") or []:
        groups[row["band"]].append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pages = []
    for band, records in sorted(groups.items()):
        records.sort(key=lambda row: row["ordinal"])
        for start in range(0, len(records), args.items_per_page):
            chunk = records[start : start + args.items_per_page]
            images = [Image.open(row["sheet"]).convert("RGB") for row in chunk]
            width = max(image.width for image in images)
            height = sum(image.height for image in images)
            page = Image.new("RGB", (width, height), (8, 8, 8))
            y = 0
            for image in images:
                page.paste(image, (0, y))
                y += image.height
            safe_band = band.replace("/", "_")
            target = args.out_dir / f"{safe_band}_p{start // args.items_per_page:02d}.jpg"
            page.save(target, quality=92)
            pages.append(
                {
                    "band": band,
                    "page": str(target),
                    "item_ids": [row["item_id"] for row in chunk],
                }
            )
    (args.out_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "instructional_shadow_band_collated_pages",
                "items_per_page": args.items_per_page,
                "pages": pages,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"bands": len(groups), "pages": len(pages)}))


if __name__ == "__main__":
    main()
