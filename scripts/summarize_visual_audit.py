#!/usr/bin/env python3
"""Summarize a completed visual-audit bundle from its manifest and review TSVs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def ratio(values: list[str], positive: str = "yes") -> str:
    count = sum(value == positive for value in values)
    return f"{count}/{len(values)} ({100 * count / len(values):.1f}%)" if values else "n/a"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--min-category-n", type=int, default=2)
    args = parser.parse_args()

    manifest = json.loads((args.bundle / "manifest.json").read_text())
    label_path = args.bundle / "manual_labels.tsv"
    if not label_path.is_file():
        label_path = args.bundle / "visual_review.tsv"
    labels = {
        int(row["audit_index"]): row
        for row in csv.DictReader(label_path.open(), delimiter="\t")
        if row.get("weak_label_valid")
    }

    print("pool_counts", json.dumps(manifest.get("pool_counts") or {}, sort_keys=True))
    print("sample_counts", json.dumps(manifest.get("sample_counts") or {}, sort_keys=True))
    for pillar in ("instructional", "witnessed"):
        rows = [row for row in manifest["visual_samples"] if row["pillar"] == pillar]
        reviewed = [(row, labels[row["audit_index"]]) for row in rows if row["audit_index"] in labels]
        print(f"{pillar}_valid", ratio([label["weak_label_valid"] for _, label in reviewed]))
        print(f"{pillar}_visual_demo", ratio([label["visual_demo_present"] for _, label in reviewed]))
        field = "polarity" if pillar == "instructional" else "query_source"
        grouped: dict[str, list[str]] = defaultdict(list)
        for row, label in reviewed:
            grouped[str(row.get(field))].append(label["weak_label_valid"])
        for key in sorted(grouped):
            print(f"  {field}={key}", ratio(grouped[key]))
        categories: dict[str, list[str]] = defaultdict(list)
        for row, label in reviewed:
            categories[str(row.get("category"))].append(label["weak_label_valid"])
        for key in sorted(categories):
            if len(categories[key]) >= args.min_category_n:
                print(f"  category={key}", ratio(categories[key]))

    commentary_path = args.bundle / "commentary_labels.tsv"
    if commentary_path.is_file():
        commentary = list(csv.DictReader(commentary_path.open(), delimiter="\t"))
        print(
            "commentary_valid",
            ratio([row["normative_statement_valid"] for row in commentary]),
        )


if __name__ == "__main__":
    main()
