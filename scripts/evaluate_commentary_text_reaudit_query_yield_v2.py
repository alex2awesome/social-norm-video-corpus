#!/usr/bin/env python3
"""Report search-stratum yield under the strict commentary occurred-event re-audit."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from evaluate_commentary_text_reaudit_v2 import file_sha256, read_jsonl, validate


def grouped(
    source_by_id: dict[str, dict[str, Any]],
    reviews: list[dict[str, Any]],
    field: str,
    minimum_action_n: int,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reviews:
        value = str(source_by_id[row["item_id"]].get(field) or "<missing>")
        groups[value].append(row)
    output: dict[str, Any] = {}
    for value, items in sorted(groups.items()):
        strict = sum(row["strict_event_label_decision"] == "accept" for row in items)
        preserved = sum(row["visual_search_route"] != "none" for row in items)
        output[value] = {
            "reviewed": len(items),
            "strict_event_labels": strict,
            "strict_event_label_rate": strict / len(items),
            "preserved_retrieval_leads": preserved,
            "preserved_retrieval_lead_rate": preserved / len(items),
            "meets_minimum_n_for_search_action": len(items) >= minimum_action_n,
            "automatic_search_action": None,
        }
    return output


def evaluate_yield(
    source_rows: list[dict[str, Any]],
    review_rows: list[dict[str, Any]],
    source_sha256: str,
    expected_source_sha256: str,
    minimum_action_n: int = 10,
) -> dict[str, Any]:
    base = validate(
        source_rows, review_rows, expected_source_sha256, source_sha256
    )
    source_by_id = {row["item_id"]: row for row in source_rows}
    strict = sum(row["strict_event_label_decision"] == "accept" for row in review_rows)
    preserved = sum(row["visual_search_route"] != "none" for row in review_rows)
    return {
        "kind": "commentary_text_reaudit_query_yield_v2",
        "reviewed_sources": len(review_rows),
        "strict_event_labels": strict,
        "strict_event_label_rate": strict / len(review_rows),
        "preserved_retrieval_leads": preserved,
        "preserved_retrieval_lead_rate": preserved / len(review_rows),
        "v1_accept_retention": base["v1_accept_retention"],
        "by_query_source": grouped(
            source_by_id, review_rows, "query_source", minimum_action_n
        ),
        "by_source_platform": grouped(
            source_by_id, review_rows, "source_platform", minimum_action_n
        ),
        "by_category": grouped(
            source_by_id, review_rows, "category", minimum_action_n
        ),
        "minimum_group_n_for_search_action": minimum_action_n,
        "automatic_search_change_authorized": False,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-gold", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--minimum-action-n", type=int, default=10)
    args = parser.parse_args()
    source_rows = read_jsonl(args.source_gold)
    review_rows = read_jsonl(args.review)
    manifest = json.loads(args.manifest.read_text())
    report = evaluate_yield(
        source_rows,
        review_rows,
        file_sha256(args.source_gold),
        str(manifest.get("source_gold_sha256") or ""),
        args.minimum_action_n,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
