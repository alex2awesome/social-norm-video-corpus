#!/usr/bin/env python3
"""Evaluate every fixed commentary crop from a completed blind manual ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


TRINARY = {"yes", "no", "uncertain"}
MEDIA = {"live_action", "animation", "dramatized", "mixed", "other", "uncertain"}
REQUIRED_YES = (
    "performed_event_visible",
    "actor_target_grounded",
    "start_boundary_clean",
    "end_boundary_clean",
    "label_bearing_text_absent",
    "audio_absent",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    result = {str(row.get("candidate_id") or ""): row for row in rows}
    if "" in result or len(result) != len(rows):
        raise ValueError(f"{name} has empty or duplicate candidate_id")
    return result


def evaluate(
    manifest: list[dict[str, Any]], blind: list[dict[str, str]],
    *, minimum_variant_rate: float = 0.8,
    minimum_parent_salvage_rate: float = 0.5,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    media = keyed(manifest, "manifest")
    reviews = keyed(blind, "blind ledger")
    if set(media) != set(reviews):
        raise ValueError("blind ledger does not exactly cover crop manifest")
    variants: dict[str, list[bool]] = {}
    parent_passes: dict[str, bool] = {}
    accepted: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    for candidate_id, item in media.items():
        review = reviews[candidate_id]
        for field in REQUIRED_YES:
            if review.get(field) not in TRINARY:
                raise ValueError(f"{candidate_id}: invalid or missing {field}")
        if review.get("medium") not in MEDIA:
            raise ValueError(f"{candidate_id}: invalid or missing medium")
        if not review.get("blind_evidence", "").strip():
            raise ValueError(f"{candidate_id}: missing blind_evidence")
        checks = {field: review[field] == "yes" for field in REQUIRED_YES}
        passed = all(checks.values())
        for field, check in checks.items():
            if not check:
                failure_counts[field] = failure_counts.get(field, 0) + 1
        variant = str(item["variant"])
        parent = str(item["parent_candidate_id"])
        variants.setdefault(variant, []).append(passed)
        parent_passes[parent] = parent_passes.get(parent, False) or passed
        if passed:
            accepted.append({
                "candidate_id": candidate_id,
                "parent_candidate_id": parent,
                "variant": variant,
                "proxy_clip": item["proxy_clip"],
                "proxy_clip_sha256": item["proxy_clip_sha256"],
                "manual_acceptance": "exact_commentary_crop_artifact",
                "automatic_acceptance": False,
                "corpus_disposition": None,
            })
    parent_count = len(parent_passes)
    salvaged = sum(parent_passes.values())
    variant_metrics = {
        variant: {
            "items": len(outcomes),
            "usable": sum(outcomes),
            "usable_rate": sum(outcomes) / len(outcomes),
        }
        for variant, outcomes in sorted(variants.items())
    }
    passing_variants = sorted(
        variant for variant, metrics in variant_metrics.items()
        if metrics["usable_rate"] >= minimum_variant_rate
    )
    parent_salvage_rate = salvaged / parent_count
    transform_candidate_promoted = bool(
        passing_variants and parent_salvage_rate >= minimum_parent_salvage_rate
    )
    report = {
        "kind": "commentary_caption_crop_manual_evaluation_v1",
        "items": len(media),
        "parents": parent_count,
        "manually_usable_outputs": len(accepted),
        "parents_salvaged_by_any_variant": salvaged,
        "parent_salvage_rate": parent_salvage_rate,
        "variant_metrics": variant_metrics,
        "failure_counts": dict(sorted(failure_counts.items())),
        "thresholds": {
            "minimum_single_fixed_variant_usable_rate": minimum_variant_rate,
            "minimum_any_variant_parent_salvage_rate": minimum_parent_salvage_rate,
        },
        "passing_fixed_variants": passing_variants,
        "shadow_transform_candidate_promoted": transform_candidate_promoted,
        "automatic_keep_rule_promoted": False,
        "oracle_per_video_variant_selection_authorized": False,
        "corpus_mutated": False,
    }
    return report, accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.accepted.exists():
        raise SystemExit("refusing to overwrite evaluation artifacts")
    report, accepted = evaluate(read_jsonl(args.manifest), read_tsv(args.blind))
    args.accepted.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted)
    )
    report["artifact_sha256"] = {
        "manifest": sha256(args.manifest),
        "blind": sha256(args.blind),
        "accepted": sha256(args.accepted),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
