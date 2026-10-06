#!/usr/bin/env python3
"""Analyze lifetime query outcomes and validate a typical-norm query proposal."""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


CONCEPT_PATTERNS = {
    "queue_line_cutting": r"queue|line cut|cutting in line|jump.*line",
    "interruption": r"interrupt|talk over|speaks over",
    "public_noise": r"loud music|speakerphone|quiet zone|library noise|noise complaint",
    "littering_shared_cleanup": r"litter|leave.*trash|shopping cart|dog poop|pick up after dog",
    "customer_worker_treatment": r"rude customer|berat.*cashier|yell.*server|customer.*employee|abuse.*staff",
    "sharing_turn_taking": r"shar(?:e|ing)|taking turns|turn taking",
    "borrowing_returning": r"borrow|return.*item|take.*without asking",
    "roommate_chores": r"roommate|housemate|chores|clean.*shared",
    "neighbor_etiquette": r"neighbor|neighbour|driveway|property line",
    "transit_seating": r"bus seat|train seat|public transport|subway etiquette|priority seat",
    "personal_space": r"personal space|too close|touch.*without",
    "phone_etiquette": r"phone etiquette|speakerphone|texting etiquette|group chat",
    "door_elevator": r"hold.*door|elevator etiquette|block.*doorway",
    "gossip_credit_exclusion": r"gossip|takes credit|steal.*credit|exclude.*coworker|left out",
    "apology_honesty": r"apolog|honesty|lying|lies to|deceiv",
    "bullying_bystander": r"bully|bystander.*interven|stand.*up for",
    "dining_guest": r"table manners|dining etiquette|dinner guest|overstay|tipping etiquette",
    "punctuality": r"punctual|late.*meeting|arriv.*late|on time",
}
OUTCOME_FIELDS = (
    "enumerated", "done", "skipped", "duplicate", "error",
    "instructional", "commentary", "witnessed", "unrouted_done",
)


def normalize_query(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value.casefold()).split())


def token_jaccard(left: str, right: str) -> float:
    a, b = set(normalize_query(left).split()), set(normalize_query(right).split())
    return len(a & b) / len(a | b) if a or b else 1.0


def read_trajectory(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        return [
            row for line in handle
            if line.strip() and (row := json.loads(line)).get("kind") == "query"
        ]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"queries": len(rows), "active": sum(bool(r["active"]) for r in rows)}
    for field in OUTCOME_FIELDS:
        result[field] = sum(int((r.get("outcomes") or {}).get(field) or 0) for r in rows)
    return result


def summarize(
    rows: list[dict[str, Any]],
    proposed: list[dict[str, Any]],
    blocked_patterns: list[str],
) -> dict[str, Any]:
    existing = {normalize_query(r["query"]): r["query"] for r in rows}
    used = [r for r in rows if r.get("last_used") is not None]
    zero_enumerated_used = [
        r for r in used if not int((r.get("outcomes") or {}).get("enumerated") or 0)
    ]
    blocked = [
        r for r in rows
        if any(re.search(pattern, r["query"], flags=re.IGNORECASE) for pattern in blocked_patterns)
    ]

    by_platform = {
        key: aggregate(group)
        for key in sorted({str(r.get("platform")) for r in rows})
        if (group := [r for r in rows if str(r.get("platform")) == key])
    }
    by_source = {
        key: aggregate(group)
        for key in sorted({str(r.get("source")) for r in rows})
        if (group := [r for r in rows if str(r.get("source")) == key])
    }
    concept_coverage = {}
    for name, pattern in CONCEPT_PATTERNS.items():
        group = [r for r in rows if re.search(pattern, r["query"], flags=re.IGNORECASE)]
        concept_coverage[name] = aggregate(group)

    proposal_rows = []
    for item in proposed:
        query = str(item["query"])
        similarities = sorted(
            ((token_jaccard(query, r["query"]), r["query"]) for r in rows),
            reverse=True,
        )
        proposal_rows.append({
            "query": query,
            "pillar": item["pillar"],
            "canary": bool(item["canary"]),
            "exact_prior_query": existing.get(normalize_query(query)),
            "blocked_pattern_matches": [
                p for p in blocked_patterns if re.search(p, query, flags=re.IGNORECASE)
            ],
            "closest_prior_query": similarities[0][1] if similarities else None,
            "closest_token_jaccard": similarities[0][0] if similarities else None,
        })

    return {
        "kind": "query_trajectory_typical_norms_v1_analysis",
        "lifetime_totals": aggregate(rows),
        "used_queries": len(used),
        "never_used_queries": len(rows) - len(used),
        "used_zero_enumerated": len(zero_enumerated_used),
        "active_used_zero_enumerated": sum(bool(r["active"]) for r in zero_enumerated_used),
        "blocked_theme": aggregate(blocked),
        "by_platform": by_platform,
        "by_source": by_source,
        "concept_coverage": concept_coverage,
        "proposal": {
            "queries": len(proposed),
            "unique_normalized": len({normalize_query(i["query"]) for i in proposed}),
            "canaries": sum(bool(i["canary"]) for i in proposed),
            "by_pillar": dict(sorted(Counter(i["pillar"] for i in proposed).items())),
            "exact_prior_query_count": sum(r["exact_prior_query"] is not None for r in proposal_rows),
            "blocked_pattern_match_count": sum(bool(r["blocked_pattern_matches"]) for r in proposal_rows),
            "rows": proposal_rows,
        },
        "limitations": [
            "Legacy rows store lifetime outcomes but not individual zero-yield passes.",
            "Query yield is retrieval evidence, not clip precision or a keep label.",
            "Proposed queries require manual output audit before expansion beyond canaries.",
        ],
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    args = parser.parse_args()
    proposed = yaml.safe_load(args.proposal.read_text())["queries"]
    settings = yaml.safe_load(args.settings.read_text())
    blocked = settings.get("scheduler", {}).get("blocked_query_patterns", [])
    print(json.dumps(
        summarize(read_trajectory(args.trajectory), proposed, blocked),
        indent=2, sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
