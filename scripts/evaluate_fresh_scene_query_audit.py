#!/usr/bin/env python3
"""Evaluate a frozen blind/reveal audit of fresh scene-oriented query outputs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _index(rows: Iterable[dict[str, Any]], name: str) -> dict[int, dict[str, Any]]:
    indexed: dict[int, dict[str, Any]] = {}
    for row in rows:
        audit_index = row.get("audit_index")
        if not isinstance(audit_index, int):
            raise ValueError(f"{name} row missing integer audit_index")
        if audit_index in indexed:
            raise ValueError(f"{name} duplicate audit_index: {audit_index}")
        indexed[audit_index] = row
    return indexed


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blind_counts = Counter(row["blind"]["any_pillar_visual_candidate"] for row in rows)
    label_counts = Counter(row["post"]["assigned_label_match"] for row in rows)
    route_counts = Counter(row["post"]["usable_route"] for row in rows)
    strict = [row for row in rows if row["post"]["strict_pillar_pass"]]
    sources = {row["sealed"]["uid"] for row in rows}
    strict_sources = {row["sealed"]["uid"] for row in strict}
    return {
        "units": len(rows),
        "sources": len(sources),
        "blind_visual_candidate": {
            key: blind_counts.get(key, 0) for key in ("yes", "uncertain", "no")
        },
        "assigned_label_match": {
            key: label_counts.get(key, 0) for key in ("yes", "uncertain", "no")
        },
        "strict_pass_units": len(strict),
        "strict_pass_unit_rate": _rate(len(strict), len(rows)),
        "strict_pass_sources": len(strict_sources),
        "strict_pass_source_rate": _rate(len(strict_sources), len(sources)),
        "routes": dict(sorted(route_counts.items())),
    }


def evaluate(
    sealed_rows: list[dict[str, Any]],
    blind_rows: list[dict[str, Any]],
    post_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    sealed = _index(sealed_rows, "sealed")
    blind = _index(blind_rows, "blind")
    post = _index(post_rows, "post")
    if set(sealed) != set(blind) or set(sealed) != set(post):
        raise ValueError(
            "audit coverage mismatch: "
            f"sealed={len(sealed)} blind={len(blind)} post={len(post)}"
        )

    joined = [
        {"sealed": sealed[index], "blind": blind[index], "post": post[index]}
        for index in sorted(sealed)
    ]
    by_modality: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_query_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        source = row["sealed"]
        by_modality[source["source_modality"]].append(row)
        by_query_category[source["query_category"]].append(row)
        by_query[source["query"]].append(row)
        by_uid[source["uid"]].append(row)

    source_contributions = []
    for uid, rows in sorted(by_uid.items()):
        strict_units = sum(row["post"]["strict_pillar_pass"] for row in rows)
        if strict_units:
            source_contributions.append(
                {
                    "uid": uid,
                    "title": rows[0]["sealed"]["title"],
                    "modality": rows[0]["sealed"]["source_modality"],
                    "audited_units": len(rows),
                    "strict_pass_units": strict_units,
                }
            )
    source_contributions.sort(
        key=lambda row: (-row["strict_pass_units"], row["uid"])
    )

    return {
        "kind": "fresh_scene_query_blind_reveal_audit",
        "overall": _summarize(joined),
        "by_modality": {
            key: _summarize(rows) for key, rows in sorted(by_modality.items())
        },
        "by_query_category": {
            key: {
                **_summarize(rows),
                "routed_modalities": dict(
                    sorted(
                        Counter(
                            row["sealed"]["source_modality"] for row in rows
                        ).items()
                    )
                ),
            }
            for key, rows in sorted(by_query_category.items())
        },
        "by_query": {
            key: {
                **_summarize(rows),
                "routed_modalities": dict(
                    sorted(
                        Counter(
                            row["sealed"]["source_modality"] for row in rows
                        ).items()
                    )
                ),
            }
            for key, rows in sorted(by_query.items())
        },
        "strict_source_contributions": source_contributions,
        "decision": {
            "automatic_promotion": False,
            "reason": (
                "Audit estimates query and weak-label precision; rules remain shadow-only "
                "until independently confirmed on source-disjoint outputs."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    report = evaluate(
        read_jsonl(args.sealed),
        read_jsonl(args.blind),
        read_jsonl(args.post),
    )
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
