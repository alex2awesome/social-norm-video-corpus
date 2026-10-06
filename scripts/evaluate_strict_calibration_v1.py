#!/usr/bin/env python3
"""Summarize a frozen strict calibration and verify exact manual coverage."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(selection: Path, blind: Path, post: Path) -> dict:
    selected = {int(row["audit_index"]): row for row in read(selection)}
    blind_rows = {int(row["audit_index"]): row for row in read(blind)}
    post_rows = {int(row["audit_index"]): row for row in read(post)}
    if not (set(selected) == set(blind_rows) == set(post_rows)):
        raise ValueError("selection, blind review, and post-reveal review differ")
    if any(post_rows[index]["pillar"] != selected[index]["pillar"] for index in selected):
        raise ValueError("post-reveal pillar mismatch")
    by_pillar = defaultdict(Counter)
    by_cell = defaultdict(Counter)
    for index, row in selected.items():
        decision = post_rows[index]["strict_decision"]
        by_pillar[row["pillar"]][decision] += 1
        by_cell[(row["pillar"], row["coverage_origin"], row["activity_band"])][decision] += 1
    return {
        "kind": "strict_audit_calibration_v1_manual_summary",
        "items": len(selected),
        "source_disjoint": len({row["uid"] for row in selected.values()}) == len(selected),
        "by_pillar": {pillar: dict(sorted(values.items())) for pillar, values in sorted(by_pillar.items())},
        "by_cell": {"|".join(key): dict(sorted(values.items())) for key, values in sorted(by_cell.items())},
        "blind_scene_visible": dict(sorted(Counter(row["situated_social_scene"] for row in blind_rows.values()).items())),
        "dense_followups": sum(row["dense_review_required"] == "yes" for row in blind_rows.values()),
        "automatic_acceptance": False,
        "automatic_rejection": False,
        "corpus_mutated": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    result = evaluate(args.selection, args.blind, args.post)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
