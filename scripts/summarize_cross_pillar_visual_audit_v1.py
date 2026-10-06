#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    pillars = defaultdict(list)
    for row in rows:
        pillars[row["pillar"]].append(row)
    out = {"kind": "cross_pillar_supervision_v1_manual_visual_summary",
           "reviewer": "gpt-5.6-sol_class_manual_visual_review",
           "sampling": "source_disjoint_deterministic_recent_batch",
           "frames_per_source": 3, "items": len(rows), "automatic_acceptance": False,
           "corpus_mutation_authorized": False, "pillars": {}}
    for pillar, items in sorted(pillars.items()):
        decisions = Counter(x["visual_target"] for x in items)
        failures = Counter(x["failure"] for x in items if x["failure"] != "none")
        out["pillars"][pillar] = {
            "items": len(items), "decisions": dict(sorted(decisions.items())),
            "confirmed_rate": decisions["yes"] / len(items),
            "confirmed_or_uncertain_rate": (decisions["yes"] + decisions["uncertain"]) / len(items),
            "failure_counts": dict(failures.most_common()),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    report = summarize(args.ledger)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["pillars"], sort_keys=True))


if __name__ == "__main__":
    main()
