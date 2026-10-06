#!/usr/bin/env python3
"""Evaluate the frozen source-disjoint WWYD staging-cue confirmation.

The broad title match remains a review-retrieval cue.  The exact official
channel rule may be promoted only for a source-level, non-destructive organic
witnessed exclusion.  It never accepts a clip as an instructional demo: that
still requires the separate visual demo contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable


TRI = {"yes", "no", "uncertain"}
GATE = {
    "minimum_selected_sources": 30,
    "minimum_precision": 0.95,
    "minimum_precision_wilson_95_lower": 0.90,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    p = successes / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total**2))
    return [(center - radius) / denominator, (center + radius) / denominator]


def validate(
    sealed: list[dict[str, Any]],
    blind: list[dict[str, str]],
    post: list[dict[str, str]],
) -> None:
    expected = [str(row["audit_index"]) for row in sealed]
    if [row["audit_index"] for row in blind] != expected:
        raise ValueError("blind ledger order/coverage mismatch")
    if [row["audit_index"] for row in post] != expected:
        raise ValueError("post-reveal ledger order/coverage mismatch")
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate audit indices")
    for row in blind:
        for field in (
            "deliberately_produced_setup",
            "contains_enacted_social_scenario",
            "instructional_demo_candidate",
        ):
            if row[field] not in TRI:
                raise ValueError(f"invalid blind {field}: {row['audit_index']}")
        if not row["visual_evidence"].strip():
            raise ValueError(f"missing blind evidence: {row['audit_index']}")
    for sealed_row, row in zip(sealed, post, strict=True):
        for field in (
            "official_wwyd_channel",
            "source_nonorganic_produced",
            "selected_clip_enacted_scenario",
            "instructional_demo_review",
        ):
            if row[field] not in {"yes", "no"}:
                raise ValueError(f"invalid post-reveal {field}: {row['audit_index']}")
        expected_official = (sealed_row.get("channel") or "").strip().casefold() == "what would you do?"
        if (row["official_wwyd_channel"] == "yes") != expected_official:
            raise ValueError(f"official-channel mismatch: {row['audit_index']}")
        if row["instructional_demo_review"] == "yes" and row["selected_clip_enacted_scenario"] != "yes":
            raise ValueError(f"demo review without enacted scene: {row['audit_index']}")
        if not row["post_reveal_evidence"].strip():
            raise ValueError(f"missing post-reveal evidence: {row['audit_index']}")


def metric(
    sealed: list[dict[str, Any]],
    post: list[dict[str, str]],
    predicate: Callable[[dict[str, Any]], bool],
    gate: dict[str, float | int] = GATE,
) -> dict[str, Any]:
    selected = [
        (source, gold)
        for source, gold in zip(sealed, post, strict=True)
        if predicate(source)
    ]
    tp = sum(gold["source_nonorganic_produced"] == "yes" for _, gold in selected)
    fp = len(selected) - tp
    interval = wilson(tp, len(selected))
    precision = tp / len(selected) if selected else None
    checks = {
        "minimum_selected_sources": len(selected) >= int(gate["minimum_selected_sources"]),
        "minimum_precision": precision is not None and precision >= float(gate["minimum_precision"]),
        "minimum_precision_wilson_95_lower": interval is not None and interval[0] >= float(gate["minimum_precision_wilson_95_lower"]),
    }
    return {
        "selected_sources": len(selected),
        "true_positive": tp,
        "false_positive": fp,
        "precision": precision,
        "precision_wilson_95": interval,
        "selected_clip_enacted_scene_rate": (
            sum(gold["selected_clip_enacted_scenario"] == "yes" for _, gold in selected) / len(selected)
            if selected
            else None
        ),
        "false_positive_item_ids": [
            source["item_id"]
            for source, gold in selected
            if gold["source_nonorganic_produced"] != "yes"
        ],
        "gate_checks": checks,
        "passes_gate": all(checks.values()),
    }


def evaluate(
    sealed: list[dict[str, Any]],
    blind: list[dict[str, str]],
    post: list[dict[str, str]],
) -> dict[str, Any]:
    validate(sealed, blind, post)
    broad = metric(sealed, post, lambda row: True)
    official = metric(
        sealed,
        post,
        lambda row: (row.get("channel") or "").strip().casefold() == "what would you do?",
    )
    return {
        "kind": "witnessed_wwyd_cue_confirmation_v1",
        "items": len(sealed),
        "source_disjoint": True,
        "blind_visual_review_complete": True,
        "post_reveal_review_complete": True,
        "gold_source_nonorganic_produced": sum(row["source_nonorganic_produced"] == "yes" for row in post),
        "gold_selected_clip_enacted_scenario": sum(row["selected_clip_enacted_scenario"] == "yes" for row in post),
        "routes": dict(sorted(Counter(row["recommended_route"] for row in post).items())),
        "gate": GATE,
        "rules": {
            "broad_wwyd_title_candidate": broad,
            "exact_official_wwyd_channel": official,
        },
        "decision": {
            "promote_broad_wwyd_title": broad["passes_gate"],
            "promote_exact_official_wwyd_channel": official["passes_gate"],
            "promoted_action": (
                "source-level strict-organic-witnessed exclusion plus instructional review priority; "
                "preserve original; never auto-accept a demo"
                if official["passes_gate"]
                else None
            ),
            "corpus_mutation": "none",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--blind-ledger", type=Path, required=True)
    parser.add_argument("--post-ledger", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    report = evaluate(
        read_jsonl(args.sealed),
        read_tsv(args.blind_ledger),
        read_tsv(args.post_ledger),
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
