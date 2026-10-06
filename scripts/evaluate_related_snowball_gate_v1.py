#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from src.snowball_scope import POLICY_VERSION, audit_snowball_parent
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.snowball_scope import POLICY_VERSION, audit_snowball_parent


def evaluate(path: Path) -> dict:
    gold = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    rows = []
    for item in gold:
        decision = audit_snowball_parent(
            title=item["title"], agent=item["agent"], scene=item["scene"],
            reaction_count=item["reaction_count"])
        rows.append({**item, "predicted_propagate": decision.allowed,
                     "predicted_reason": decision.reason,
                     "correct": decision.allowed == item["manual_propagate"]})
    tp = sum(x["manual_propagate"] and x["predicted_propagate"] for x in rows)
    tn = sum(not x["manual_propagate"] and not x["predicted_propagate"] for x in rows)
    fp = sum(not x["manual_propagate"] and x["predicted_propagate"] for x in rows)
    fn = sum(x["manual_propagate"] and not x["predicted_propagate"] for x in rows)
    return {"kind": "related_snowball_parent_gate_v1_manual_evaluation",
            "policy_version": POLICY_VERSION, "items": len(rows),
            "true_allow": tp, "true_block": tn, "false_allow": fp,
            "false_block": fn, "accuracy": (tp + tn) / len(rows),
            "promotion_gate": {"minimum_items": 20, "requires_zero_false_allows": True,
                               "requires_at_least_one_true_allow": True,
                               "passed": len(rows) >= 20 and fp == 0 and tp >= 1},
            "scope": "future_related_query_propagation_only",
            "delete_media": False, "corpus_mutation_authorized": False,
            "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = evaluate(args.ledger)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: report[k] for k in ("items", "true_allow", "true_block", "false_allow", "false_block", "accuracy")}))


if __name__ == "__main__":
    main()
