#!/usr/bin/env python3
"""Validate and summarize the exhaustive commentary capture-title audit.

This combines the existing moving-video adjudication of the 60 blind-uncertain
sources with a sealed manual source-level adjudication of the 94 blind-clear
sources. It writes only derived shadow-audit artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROUTES = (
    "exact_visual_localization_review",
    "relabel_visual_localization_review",
    "instructional_demo_review",
    "reject_no_visible_social_event",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def route_uncertain(row: dict[str, str]) -> str:
    if row["usable_weak_supervision"] != "yes":
        return "reject_no_visible_social_event"
    if row["manual_class"] == "instructional_reroute":
        return "instructional_demo_review"
    if row["label_alignment"] == "same_norm":
        return "relabel_visual_localization_review"
    if row["label_alignment"] == "exact":
        return "exact_visual_localization_review"
    raise ValueError(f"Usable row has unsupported alignment: {row}")


def evaluate(
    semantic_path: Path,
    blind_path: Path,
    uncertain_path: Path,
    clear_path: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    semantic = {
        int(row["audit_index"]): row
        for row in map(json.loads, semantic_path.read_text().splitlines())
        if row
    }
    blind = {int(row["audit_index"]): row for row in read_tsv(blind_path)}
    uncertain_rows = read_tsv(uncertain_path)
    uncertain = {int(row["audit_index"]): row for row in uncertain_rows}
    clear = json.loads(clear_path.read_text())

    if set(semantic) != set(range(154)) or set(blind) != set(range(154)):
        raise ValueError("Semantic and blind ledgers must each cover indices 0..153")
    if len(uncertain) != 60 or set(uncertain) != {
        index for index, row in blind.items() if row["visual_status"] == "U"
    }:
        raise ValueError("Moving-video ledger must cover all and only 60 blind-U rows")

    clear_routes: dict[int, str] = {}
    for route in ROUTES:
        for raw_index in clear[route]:
            index = int(raw_index)
            if index in clear_routes:
                raise ValueError(f"Clear index appears in two routes: {index}")
            clear_routes[index] = route
    expected_clear = set(blind) - set(uncertain)
    if set(clear_routes) != expected_clear:
        missing = sorted(expected_clear - set(clear_routes))
        extra = sorted(set(clear_routes) - expected_clear)
        raise ValueError(f"Clear adjudication partition mismatch: missing={missing}, extra={extra}")
    for index, route in clear_routes.items():
        status = blind[index]["visual_status"]
        if status == "N" and route != "reject_no_visible_social_event":
            raise ValueError(f"Blind-N source routed positive: {index}")
        if status == "Y" and route == "reject_no_visible_social_event":
            raise ValueError(f"Blind-Y source routed negative: {index}")

    evaluated: list[dict[str, object]] = []
    for index in range(154):
        moving = uncertain.get(index)
        route = route_uncertain(moving) if moving else clear_routes[index]
        label_alignment = (
            moving["label_alignment"]
            if moving
            else (
                "exact"
                if route in {"exact_visual_localization_review", "instructional_demo_review"}
                else "same_norm"
                if route == "relabel_visual_localization_review"
                else "not_usable"
            )
        )
        evaluated.append(
            {
                "audit_index": index,
                "candidate_id": blind[index]["candidate_id"],
                "uid": semantic[index]["uid"],
                "title": semantic[index]["norm"],
                "blind_visual_status": blind[index]["visual_status"],
                "route": route,
                "source_usable_candidate": route != "reject_no_visible_social_event",
                "label_alignment": label_alignment,
                "manual_evidence": (
                    moving["literal_visual_evidence"]
                    if moving
                    else blind[index]["literal_description"]
                ),
                "adjudication_provenance": (
                    "moving_video_post_reveal_ledger"
                    if moving
                    else "clear_source_post_reveal_adjudication"
                ),
                "requires_exact_clip_audit": route != "reject_no_visible_social_event",
            }
        )

    route_counts = Counter(str(row["route"]) for row in evaluated)
    visual_status_counts = Counter(str(row["blind_visual_status"]) for row in evaluated)
    usable = sum(bool(row["source_usable_candidate"]) for row in evaluated)
    summary: dict[str, object] = {
        "kind": "commentary_capture_title_exhaustive_manual_source_evaluation",
        "policy": "shadow_source_routing_only_no_corpus_mutation",
        "population": len(evaluated),
        "coverage_complete": len(evaluated) == 154,
        "blind_visual_status_counts": dict(sorted(visual_status_counts.items())),
        "route_counts": dict(sorted(route_counts.items())),
        "source_usable_candidates": usable,
        "source_usable_rate": usable / len(evaluated),
        "strict_exact_title_event_candidates": route_counts[
            "exact_visual_localization_review"
        ],
        "strict_exact_title_event_rate": route_counts[
            "exact_visual_localization_review"
        ]
        / len(evaluated),
        "requires_exact_clip_audit": usable,
        "inputs": {
            str(path.name): sha256(path)
            for path in (semantic_path, blind_path, uncertain_path, clear_path)
        },
    }
    return evaluated, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--uncertain", type=Path, required=True)
    parser.add_argument("--clear", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    args = parser.parse_args()

    evaluated, summary = evaluate(
        args.semantic, args.blind, args.uncertain, args.clear
    )
    args.out_jsonl.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in evaluated)
    )
    summary["outputs"] = {args.out_jsonl.name: sha256(args.out_jsonl)}
    args.out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
