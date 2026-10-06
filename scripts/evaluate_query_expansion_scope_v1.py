#!/usr/bin/env python3
"""Evaluate the automatic query gate against frozen manual scope judgments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

if __package__:
    from src.query_scope import POLICY_VERSION, audit_query_scope
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.query_scope import POLICY_VERSION, audit_query_scope


def evaluate(positive_yaml: Path, negative_jsonl: Path) -> dict:
    positives = yaml.safe_load(positive_yaml.read_text())["queries"]
    gold = [
        {"query": row["query"], "manual_allowed": True,
         "manual_reason": "ordinary_social_query", "provenance": "prior_54_query_manual_audit"}
        for row in positives
    ]
    gold.extend(json.loads(line) for line in negative_jsonl.read_text().splitlines() if line.strip())
    rows = []
    for item in gold:
        decision = audit_query_scope(item["query"], "llm_expand")
        rows.append({**item, "predicted_allowed": decision.allowed,
                     "predicted_reason": decision.reason,
                     "correct": decision.allowed == item["manual_allowed"]})
    tp = sum(r["manual_allowed"] and r["predicted_allowed"] for r in rows)
    tn = sum(not r["manual_allowed"] and not r["predicted_allowed"] for r in rows)
    fp = sum(not r["manual_allowed"] and r["predicted_allowed"] for r in rows)
    fn = sum(r["manual_allowed"] and not r["predicted_allowed"] for r in rows)
    return {
        "kind": "query_expansion_scope_v1_manual_evaluation",
        "policy_version": POLICY_VERSION,
        "items": len(rows), "allowed_gold": tp + fn, "blocked_gold": tn + fp,
        "true_allow": tp, "true_block": tn, "false_allow": fp, "false_block": fn,
        "accuracy": (tp + tn) / len(rows),
        "promotion_gate": {"minimum_items": 80, "requires_zero_false_blocks": True,
                           "requires_zero_false_allows": True,
                           "passed": len(rows) >= 80 and fp == 0 and fn == 0},
        "automatic_acceptance": False, "corpus_mutation_authorized": False,
        "rows": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--positive-yaml", type=Path, default=Path("config/typical_social_norm_queries_v1.yaml"))
    ap.add_argument("--negative-jsonl", type=Path, default=Path("audit_runs/20260807_query_expansion_scope_v1/manual_negative_scope_audit.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = evaluate(args.positive_yaml, args.negative_jsonl)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("items", "true_allow", "true_block", "false_allow", "false_block", "accuracy")}))


if __name__ == "__main__":
    main()
