#!/usr/bin/env python3
"""Report commentary search strata after complete manual text eligibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ACCEPTED = {"accept", "accept_after_relabel"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def grouped(rows: list[dict[str, Any]], field: str, minimum_action_n: int) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "<missing>")].append(row)
    output = {}
    for value, items in sorted(groups.items()):
        eligible = sum(row["decision"] in ACCEPTED for row in items)
        exact = sum(row["decision"] == "accept" for row in items)
        output[value] = {
            "reviewed": len(items),
            "eligible_after_manual_relabel": eligible,
            "eligible_after_manual_relabel_rate": eligible / len(items),
            "raw_detector_phrase_accepted_exact": exact,
            "raw_detector_phrase_precision": exact / len(items),
            "meets_minimum_n_for_search_action": len(items) >= minimum_action_n,
            "automatic_search_action": None,
        }
    return output


def evaluate(rows: list[dict[str, Any]], minimum_action_n: int = 10) -> dict[str, Any]:
    if not rows:
        raise ValueError("commentary text gold is empty")
    ids = [str(row.get("item_id") or "") for row in rows]
    uids = [str(row.get("uid") or "") for row in rows]
    if any(not value for value in ids + uids) or len(ids) != len(set(ids)) or len(uids) != len(set(uids)):
        raise ValueError("commentary text gold must be source-disjoint with unique item ids")
    for row in rows:
        if row.get("decision") not in ACCEPTED | {"reject"}:
            raise ValueError(f"{row['item_id']}: invalid or incomplete decision")
        if row.get("text_label_manual_reviewed") is not True:
            raise ValueError(f"{row['item_id']}: manual-review flag missing")
    eligible = sum(row["decision"] in ACCEPTED for row in rows)
    exact = sum(row["decision"] == "accept" for row in rows)
    return {
        "kind": "commentary_text_query_yield_v1",
        "reviewed_sources": len(rows),
        "eligible_for_visual_search_after_manual_relabel": eligible,
        "eligible_for_visual_search_rate": eligible / len(rows),
        "raw_detector_phrase_accepted_exact": exact,
        "raw_detector_phrase_precision": exact / len(rows),
        "by_query_source": grouped(rows, "query_source", minimum_action_n),
        "by_source_platform": grouped(rows, "source_platform", minimum_action_n),
        "by_category": grouped(rows, "category", minimum_action_n),
        "minimum_group_n_for_search_action": minimum_action_n,
        "text_eligibility_is_not_visual_certification": True,
        "automatic_search_change_authorized": False,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minimum-action-n", type=int, default=10)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = evaluate(read_jsonl(args.gold), args.minimum_action_n)
    report["gold_sha256"] = hashlib.sha256(args.gold.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
