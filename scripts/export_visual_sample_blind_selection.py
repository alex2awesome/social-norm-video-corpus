#!/usr/bin/env python3
"""Convert a visual_audit_sample manifest into a sealed/blind clip selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert(manifest: dict, root: Path) -> tuple[list[dict], list[dict]]:
    sealed = []
    blind = []
    for source in manifest.get("visual_samples") or []:
        index = int(source["audit_index"])
        uid = str(source["uid"])
        item_id = f"{source['pillar']}_source:{uid}"
        clip = Path(source["clip_path"])
        if not clip.is_absolute():
            clip = root / clip
        base = {
            "audit_index": index,
            "item_id": item_id,
            "uid": uid,
            "clip": str(clip),
        }
        blind.append(base)
        sealed.append({**base, **source})
    return sealed, blind


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    args.out.mkdir(parents=True)
    sealed, blind = convert(json.loads(args.manifest.read_text()), args.root)
    for name, rows in (
        ("sealed_selection.jsonl", sealed),
        ("blind_selection.jsonl", blind),
    ):
        (args.out / name).write_text(
            "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
        )
    (args.out / "preregistration.json").write_text(
        json.dumps(
            {
                "kind": "visual_sample_blind_selection",
                "items": len(sealed),
                "semantic_fields_hidden_during_visual_review": True,
                "source_manifest": str(args.manifest),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(json.dumps({"items": len(sealed), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
