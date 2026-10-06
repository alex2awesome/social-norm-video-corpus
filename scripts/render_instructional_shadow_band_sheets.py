#!/usr/bin/env python3
"""Render dense labeled sheets for manual audit of instructional score bands."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def shorten(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_frames(
    clip: Path, target: Path, duration: float, count: int, ffmpeg: str
) -> list[Path]:
    fps = count / max(duration, 0.001)
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-i",
            str(clip),
            "-vf",
            f"fps={fps:.6f},scale=320:180:force_original_aspect_ratio=decrease",
            "-frames:v",
            str(count),
            str(target / "%03d.jpg"),
        ],
        check=True,
    )
    return sorted(target.glob("*.jpg"))


def evidence(row: dict[str, Any], model: str) -> str:
    result = row.get(f"{model}_result") or {}
    return str(result.get("evidence") or "")


def render_one(
    source: dict[str, Any],
    band: dict[str, Any],
    manifest_dir: Path,
    out_dir: Path,
    count: int,
    ffmpeg: str,
) -> dict[str, Any]:
    clip = Path(source["proxy_clip"])
    local_clip = manifest_dir / "clips" / clip.name
    if local_clip.is_file():
        clip = local_clip
    target = out_dir / f"{int(source['ordinal']):04d}_{source['uid']}.jpg"
    if not target.exists():
        with tempfile.TemporaryDirectory(prefix="shadow-band-frames-") as temp:
            frames = extract_frames(
                clip,
                Path(temp),
                float(source.get("duration") or source.get("duration_hint") or 1),
                count,
                ffmpeg,
            )
            columns = 4
            rows = (count + columns - 1) // columns
            header = 150
            sheet = Image.new("RGB", (columns * 320, header + rows * 180), (18, 18, 18))
            draw = ImageDraw.Draw(sheet)
            font = ImageFont.load_default()
            labels = [
                f"{source['item_id']} | {band['band']} | {float(source.get('duration') or 0):.1f}s",
                f"polarity={source.get('polarity')} | category={source.get('category')} | norm={shorten(source.get('norm'), 110)}",
                f"Qwen: {shorten(evidence(band, 'qwen'), 180)}",
                f"GLM: {shorten(evidence(band, 'glm'), 180)}",
                f"Text: {shorten((band.get('text_result') or {}).get('reason'), 180)}",
            ]
            for index, label in enumerate(labels):
                draw.text((6, 6 + index * 27), label, fill="white", font=font)
            for index, frame in enumerate(frames):
                image = Image.open(frame).convert("RGB")
                x = (index % columns) * 320 + (320 - image.width) // 2
                y = header + (index // columns) * 180 + (180 - image.height) // 2
                sheet.paste(image, (x, y))
            sheet.save(target, quality=90)
    return {
        "ordinal": source["ordinal"],
        "item_id": source["item_id"],
        "uid": source["uid"],
        "band": band["band"],
        "sheet": str(target),
        "sheet_sha256": file_sha256(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--bands", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--band", action="append", default=[])
    args = parser.parse_args()
    if args.out_manifest.exists():
        raise SystemExit(f"refusing to overwrite: {args.out_manifest}")

    sources = load_jsonl(args.manifest)
    bands = {row["item_id"]: row for row in load_jsonl(args.bands)}
    if set(bands) != {row["item_id"] for row in sources}:
        raise SystemExit("band/source identities are not exact")
    if args.band:
        sources = [row for row in sources if bands[row["item_id"]]["band"] in args.band]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                render_one,
                row,
                bands[row["item_id"]],
                args.manifest.parent,
                args.out_dir,
                args.frames,
                args.ffmpeg,
            ): row
            for row in sources
        }
        for index, future in enumerate(as_completed(futures), 1):
            records.append(future.result())
            if index % 25 == 0:
                print(f"{index}/{len(futures)}", flush=True)
    records.sort(key=lambda row: row["ordinal"])
    args.out_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "instructional_shadow_band_manual_sheets",
                "items": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"items": len(records), "out": str(args.out_manifest)}))


if __name__ == "__main__":
    main()
