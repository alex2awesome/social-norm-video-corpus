#!/usr/bin/env python3
"""Score human 2AFC judgments on the matched reaction pairs.

Joins exported page judgments with the ground-truth pair manifest (which the
page never contained), reports human accuracy with Wilson bounds, sure-vs-
guessing breakdown, flag rates (the pair-quality audit), and the head-to-head
with the VLM on the pair subset both saw.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from weaksup.lf_matrix_v1 import wilson_lower

SCORER_VERSION = "score_human_pairs_v1"


def score(pairs: list[dict[str, Any]], judgments: list[dict[str, Any]]) -> dict[str, Any]:
    truth = {p["pair_id"]: p for p in pairs}
    unknown = [j["pair_id"] for j in judgments if j["pair_id"] not in truth]
    if unknown:
        raise ValueError(f"judgments reference unknown pairs: {unknown[:3]}")
    decided = [j for j in judgments if j.get("choice") in ("A", "B")]
    cant_tell = [j for j in judgments if j.get("choice") == "cant_tell"]
    correct = [j for j in decided
               if j["choice"] == truth[j["pair_id"]]["positive_side"]]
    sure = [j for j in decided if j.get("confidence") == "sure"]
    sure_correct = [j for j in sure
                    if j["choice"] == truth[j["pair_id"]]["positive_side"]]
    flags: dict[str, int] = {}
    for j in judgments:
        for f in j.get("flags") or []:
            flags[f] = flags.get(f, 0) + 1
    vlm_pairs = [p for p in pairs if p.get("vlm_choice_correct") is not None]
    vlm_ids = {p["pair_id"] for p in vlm_pairs}
    overlap = [j for j in decided if j["pair_id"] in vlm_ids]
    overlap_correct = [j for j in overlap
                       if j["choice"] == truth[j["pair_id"]]["positive_side"]]
    return {
        "scorer_version": SCORER_VERSION,
        "pairs_total": len(pairs), "judged": len(decided) + len(cant_tell),
        "decided": len(decided), "cant_tell": len(cant_tell),
        "human_accuracy": len(correct) / len(decided) if decided else None,
        "human_accuracy_wilson_lower": wilson_lower(len(correct), len(decided)),
        "sure_n": len(sure),
        "sure_accuracy": len(sure_correct) / len(sure) if sure else None,
        "flag_counts": flags,
        "vlm_headtohead": {
            "pairs_both_saw": len(vlm_pairs),
            "human_decided_on_those": len(overlap),
            "human_correct": len(overlap_correct),
            "vlm_correct": sum(1 for p in vlm_pairs if p["vlm_choice_correct"]),
        },
        "chance_is": 0.5,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-manifest", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    pairs = json.loads(args.pairs_manifest.read_text())
    judgments = [json.loads(l) for l in args.judgments.read_text().splitlines() if l.strip()]
    report = score(pairs, judgments)
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
