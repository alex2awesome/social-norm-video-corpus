#!/usr/bin/env python3
"""Export a fail-closed VLM manifest containing only OCR-masked pages.

The development renderer intentionally keeps paired unmasked pages for human
audit.  This exporter creates the separate artifact that may be copied to a
model host.  It refuses absolute paths, non-masked page roots, missing hashes,
and any retained key or string containing ``unmasked``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any


HUMAN_ONLY_KEYS = {
    "human_unmasked_page_paths",
    "human_unmasked_page_sha256",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    model_row = {key: value for key, value in row.items() if key not in HUMAN_ONLY_KEYS}
    if model_row.get("ocr_masked") is not True:
        raise ValueError("refusing to export a row that is not explicitly OCR-masked")

    page_paths = model_row.get("page_paths")
    hashes = model_row.get("page_sha256")
    if not isinstance(page_paths, list) or not page_paths:
        raise ValueError("each row must contain at least one masked page")
    if not isinstance(hashes, list) or len(hashes) != len(page_paths):
        raise ValueError("each masked page must have one SHA-256 digest")

    for raw_path in page_paths:
        if not isinstance(raw_path, str):
            raise ValueError("page paths must be strings")
        path = PurePosixPath(raw_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe page path: {raw_path}")
        if not path.parts or path.parts[0] != "pages_ocr_masked":
            raise ValueError(f"non-masked page root: {raw_path}")
    for digest in hashes:
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"invalid SHA-256 digest: {digest!r}")

    serialized = json.dumps(model_row, sort_keys=True)
    if "unmasked" in serialized.lower():
        raise ValueError("unmasked data survived model-manifest sanitization")
    return model_row


def export_manifest(source: Path, destination: Path) -> int:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing export: {destination}")
    rows = read_jsonl(source)
    if not rows:
        raise ValueError("source manifest is empty")
    sanitized = [sanitize_row(row) for row in rows]
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        for row in sanitized:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    return len(sanitized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    count = export_manifest(args.input, args.output)
    print(json.dumps({"rows": count, "output": str(args.output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
