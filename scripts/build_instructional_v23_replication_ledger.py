#!/usr/bin/env python3
"""Join frozen visual and post-reveal audits into a V23 evaluation ledger."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any


FIELDS = [
    "audit_index", "candidate_id", "item_id", "uid", "visual_form",
    "visual_demo", "semantic_alignment", "complete_demo", "usable_demo", "note",
]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def build(
    manifest: list[dict[str, Any]],
    visual_rows: list[dict[str, str]],
    semantic_rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    visual = {int(row["audit_index"]): row for row in visual_rows}
    semantics = {int(row["audit_index"]): row for row in semantic_rows}
    by_index = {int(row["audit_index"]): row for row in manifest}
    if len(by_index) != len(manifest) or len(visual) != len(visual_rows):
        raise ValueError("duplicate audit_index")
    if set(by_index) != set(visual):
        raise ValueError("manifest/visual coverage mismatch")
    visual_positive = {
        index for index, row in visual.items()
        if row["visual_demo"].strip().upper() == "Y"
    }
    if set(semantics) != visual_positive:
        raise ValueError("semantic review must cover exactly the visual positives")
    output = []
    for index in sorted(by_index):
        source = by_index[index]
        v = visual[index]
        is_visual = index in visual_positive
        semantic = semantics.get(index)
        status = semantic["semantic_status"].strip() if semantic else ""
        exact = status == "exact"
        if semantic and status not in {"exact", "relabel", "reject_semantic"}:
            raise ValueError(f"audit_index {index}: invalid semantic_status")
        output.append({
            "audit_index": str(index),
            "candidate_id": str(source["candidate_id"]),
            "item_id": str(source["item_id"]),
            "uid": str(source["uid"]),
            "visual_form": v["visual_form"],
            "visual_demo": "yes" if is_visual else "no",
            "semantic_alignment": "yes" if exact else "no",
            "complete_demo": "yes" if is_visual else "no",
            "usable_demo": "yes" if exact else "no",
            "note": semantic["note"] if semantic else v["note"],
        })
    return output


def main() -> int:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-ledger", type=Path, required=True)
    parser.add_argument("--semantic-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = [
        json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()
    ]
    rows = build(
        manifest, read_tsv(args.visual_ledger), read_tsv(args.semantic_review)
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
