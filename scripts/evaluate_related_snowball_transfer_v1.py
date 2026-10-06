#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import json
from math import sqrt
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, center - half), min(1.0, center + half)]


def evaluate(path: Path) -> dict:
    rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    bands = {}
    for band in ("allow", "block"):
        subset = [x for x in rows if x["band"] == band]
        counts = Counter(x["strict_witnessed"] for x in subset)
        salvage = Counter(x["salvage"] for x in subset)
        bands[band] = {"items": len(subset), "decisions": dict(counts),
                       "confirmed_strict_rate": counts["yes"] / len(subset),
                       "confirmed_strict_wilson_95": wilson(counts["yes"], len(subset)),
                       "salvage": dict(salvage)}
    allowed = bands["allow"]
    return {"kind": "related_snowball_parent_gate_v1_transfer",
            "items": len(rows), "bands": bands,
            "parent_gate_transfer_passed": allowed["items"] >= 20 and allowed["decisions"].get("yes", 0) > 0,
            "recommended_action": "quarantine_dailymotion_related_queries"
                if allowed["items"] >= 20 and allowed["decisions"].get("yes", 0) == 0
                else "continue_shadow",
            "reason": "zero confirmed strict witnessed children in the proposed allow band",
            "existing_media_preserved": True, "delete_media": False,
            "corpus_mutation_authorized": False, "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = evaluate(args.ledger)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"bands": report["bands"], "recommended_action": report["recommended_action"]}, sort_keys=True))


if __name__ == "__main__":
    main()
