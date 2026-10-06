#!/usr/bin/env python3
"""Compare frozen blind occurred-event gold with historical commentary decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from scripts.select_commentary_occurred_event_transfer_v3 import ACCEPTED, read_jsonl
else:
    from select_commentary_occurred_event_transfer_v3 import ACCEPTED, read_jsonl


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [center - margin, center + margin]


def grouped(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "<missing>")].append(row)
    output = {}
    for value, items in sorted(groups.items()):
        strict = sum(item["gold_strict"] for item in items)
        prior = sum(item["prior_accept"] for item in items)
        output[value] = {
            "reviewed": len(items),
            "strict_gold": strict,
            "strict_gold_rate": strict / len(items),
            "prior_accepted": prior,
            "automatic_action": None,
        }
    return output


def evaluate(
    selection: list[dict[str, Any]],
    blind_gold: list[dict[str, Any]],
    observed_gold_sha256: str,
    expected_gold_sha256: str,
) -> dict[str, Any]:
    if observed_gold_sha256 != expected_gold_sha256:
        raise ValueError("blind manual gold hash differs from frozen hash")
    selected = {row["transfer_index"]: row for row in selection}
    gold = {row["transfer_index"]: row for row in blind_gold}
    if len(selected) != len(selection) or len(gold) != len(blind_gold) or set(selected) != set(gold):
        raise ValueError("selection and blind gold do not have exact unique index coverage")
    joined = []
    for index in sorted(selected):
        prior = selected[index]
        truth = gold[index]
        if truth.get("blind_id") != f"commentary-transfer-v3-{index:04d}":
            raise ValueError(f"index {index}: blind ID lineage mismatch")
        prior_accept = prior["v1_decision"] in ACCEPTED
        gold_strict = truth["route"] == "strict_visual_search"
        joined.append({
            **prior,
            "blind_id": truth["blind_id"],
            "prior_accept": prior_accept,
            "gold_strict": gold_strict,
            "gold_route": truth["route"],
            "gold_behavior": truth.get("normalized_behavior") or "",
            "gold_norm": truth.get("normalized_norm") or "",
            "gold_description": truth.get("description") or "",
        })
    tp = sum(row["prior_accept"] and row["gold_strict"] for row in joined)
    fp = sum(row["prior_accept"] and not row["gold_strict"] for row in joined)
    fn = sum(not row["prior_accept"] and row["gold_strict"] for row in joined)
    tn = sum(not row["prior_accept"] and not row["gold_strict"] for row in joined)
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    disagreements = [
        {
            "transfer_index": row["transfer_index"],
            "blind_id": row["blind_id"],
            "item_id": row["item_id"],
            "uid": row["uid"],
            "prior_decision": row["v1_decision"],
            "prior_behavior": row["v1_normalized_behavior"],
            "prior_norm": row["v1_normalized_norm"],
            "gold_route": row["gold_route"],
            "gold_behavior": row["gold_behavior"],
            "gold_norm": row["gold_norm"],
            "gold_description": row["gold_description"],
        }
        for row in joined
        if row["prior_accept"] != row["gold_strict"]
    ]
    return {
        "kind": "commentary_occurred_event_v3_historical_transfer_evaluation",
        "reviewed": len(joined),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "precision_wilson_95": wilson(tp, tp + fp),
        "recall": recall,
        "recall_wilson_95": wilson(tp, tp + fn),
        "strict_gold": tp + fn,
        "strict_gold_rate": (tp + fn) / len(joined),
        "by_prior_decision": grouped(joined, "v1_decision"),
        "by_source_platform": grouped(joined, "source_platform"),
        "by_query_source": grouped(joined, "query_source"),
        "disagreements": disagreements,
        "blind_gold_sha256": observed_gold_sha256,
        "model_v3_evaluated": False,
        "automatic_acceptance": False,
        "automatic_search_change_authorized": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--blind-gold", type=Path, required=True)
    parser.add_argument("--expected-gold-sha256", required=True)
    args = parser.parse_args()
    report = evaluate(
        read_jsonl(args.selection),
        read_jsonl(args.blind_gold),
        sha256(args.blind_gold),
        args.expected_gold_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
