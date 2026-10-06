#!/usr/bin/env python3
"""Build one auditable gold ledger from overlapping manual review files.

Repeated judgments with the same decision are retained as provenance.  Any
decision conflict is an error unless an explicit adjudication is supplied.
This prevents an evaluator from silently choosing whichever manual file was
listed last.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def indexed_rows(paths: list[Path]) -> dict[str, list[dict]]:
    by_item: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        for line_number, row in enumerate(load_jsonl(path), 1):
            record = dict(row)
            record["_source_file"] = path.name
            record["_source_line"] = line_number
            by_item[row["item_id"]].append(record)
    return by_item


def load_adjudications(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    result: dict[str, dict] = {}
    for row in rows:
        item_id = row["item_id"]
        if item_id in result:
            raise ValueError(f"duplicate adjudication: {item_id}")
        if not row.get("adjudication_reason"):
            raise ValueError(f"adjudication missing reason: {item_id}")
        result[item_id] = row
    return result


def consolidate(paths: list[Path], adjudication_path: Path | None = None) -> list[dict]:
    by_item = indexed_rows(paths)
    adjudications = load_adjudications(adjudication_path)
    unknown = set(adjudications) - set(by_item)
    if unknown:
        raise ValueError(f"adjudications have no source review: {sorted(unknown)}")

    output = []
    for item_id, reviews in sorted(by_item.items()):
        decisions = {row.get("decision") for row in reviews}
        if None in decisions:
            raise ValueError(f"manual review missing decision: {item_id}")
        adjudication = adjudications.get(item_id)
        if len(decisions) > 1 and adjudication is None:
            raise ValueError(
                f"conflicting decisions require adjudication: {item_id} "
                f"{sorted(decisions)}"
            )

        if adjudication is not None:
            chosen = dict(adjudication)
            adjudicated = True
        else:
            # The decision is identical. Prefer the record with the most fields,
            # while retaining every review below as provenance.
            chosen = dict(max(reviews, key=lambda row: (len(row), row["_source_file"])))
            chosen.pop("_source_file", None)
            chosen.pop("_source_line", None)
            adjudicated = False

        chosen["item_id"] = item_id
        chosen["adjudicated"] = adjudicated
        chosen["source_reviews"] = [
            {
                "file": row["_source_file"],
                "line": row["_source_line"],
                "decision": row["decision"],
                "evidence": row.get("evidence"),
            }
            for row in reviews
        ]
        output.append(chosen)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-review", type=Path, action="append", required=True)
    parser.add_argument("--adjudications", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = consolidate(args.manual_review, args.adjudications)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "items": len(rows),
                "adjudicated": sum(row["adjudicated"] for row in rows),
                "source_files": [str(path) for path in args.manual_review],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
