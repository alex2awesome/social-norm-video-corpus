#!/usr/bin/env python3
"""Evaluate adaptive commentary edge crops, including every abstention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_commentary_clip_plan import (
        ALIGNMENTS,
        MEDIA,
        ROUTES,
        TRINARY,
        keyed,
        read_jsonl,
        read_tsv,
        sha256,
    )
else:
    from evaluate_commentary_clip_plan import (
        ALIGNMENTS,
        MEDIA,
        ROUTES,
        TRINARY,
        keyed,
        read_jsonl,
        read_tsv,
        sha256,
    )


BLIND_YES = (
    "performed_event_visible",
    "actor_target_grounded",
    "start_boundary_clean",
    "end_boundary_clean",
    "label_bearing_text_absent",
    "editorial_target_marker_absent",
    "audio_absent",
)


def evaluate(
    manifest: list[dict[str, Any]],
    blind: list[dict[str, str]],
    post: list[dict[str, str]],
    materialization: dict[str, Any],
    contract: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    media = keyed(manifest, "manifest")
    visuals = keyed(blind, "blind ledger")
    semantics = keyed(post, "post-reveal ledger")
    if not (set(media) == set(visuals) == set(semantics)):
        raise ValueError("manual ledgers do not exactly cover rendered manifest")
    cohort = int(materialization["cohort_items"])
    rendered = int(materialization["rendered_items"])
    abstentions = list(materialization["abstentions"])
    if rendered != len(media):
        raise ValueError("rendered count does not match manifest")
    if cohort != rendered + len(abstentions):
        raise ValueError("cohort does not equal rendered items plus abstentions")

    accepted: list[dict[str, Any]] = []
    failures: dict[str, int] = {}
    for candidate_id, item in media.items():
        visual = visuals[candidate_id]
        semantic = semantics[candidate_id]
        for field in BLIND_YES:
            if visual.get(field) not in TRINARY:
                raise ValueError(f"{candidate_id}: invalid or missing {field}")
        if visual.get("medium") not in MEDIA:
            raise ValueError(f"{candidate_id}: invalid or missing medium")
        if not visual.get("blind_evidence", "").strip():
            raise ValueError(f"{candidate_id}: missing blind evidence")
        if semantic.get("exact_named_action_visible") not in TRINARY:
            raise ValueError(f"{candidate_id}: invalid exact action judgment")
        if semantic.get("label_alignment") not in ALIGNMENTS:
            raise ValueError(f"{candidate_id}: invalid label alignment")
        if semantic.get("route") not in ROUTES:
            raise ValueError(f"{candidate_id}: invalid route")
        if not semantic.get("post_reveal_evidence", "").strip():
            raise ValueError(f"{candidate_id}: missing post-reveal evidence")
        checks = {field: visual[field] == "yes" for field in BLIND_YES}
        checks.update({
            "exact_named_action_visible": semantic["exact_named_action_visible"] == "yes",
            "label_alignment_exact": semantic["label_alignment"] == "exact",
            "commentary_visual_route": semantic["route"] == "commentary_visual",
        })
        for name, passed in checks.items():
            if not passed:
                failures[name] = failures.get(name, 0) + 1
        if all(checks.values()):
            accepted.append({
                "candidate_id": candidate_id,
                "parent_candidate_id": item["parent_candidate_id"],
                "proxy_clip": item["proxy_clip"],
                "proxy_clip_sha256": item["proxy_clip_sha256"],
                "manual_acceptance": "exact_commentary_adaptive_crop_artifact",
                "automatic_acceptance": False,
                "corpus_disposition": None,
            })

    gate = contract["gate"]
    rendered_rate = len(accepted) / rendered if rendered else 0.0
    cohort_rate = len(accepted) / cohort
    checks = {
        "all_rendered_outputs_manually_reviewed": len(visuals) == rendered,
        "minimum_rendered_outputs": rendered >= int(gate["minimum_rendered_outputs"]),
        "minimum_usable_rate": rendered_rate >= float(gate["minimum_usable_rate"]),
        "minimum_parent_salvage_rate": cohort_rate >= float(gate["minimum_parent_salvage_rate"]),
    }
    promoted = all(checks.values())
    report = {
        "kind": "commentary_adaptive_edge_crop_manual_evaluation_v1",
        "cohort_items": cohort,
        "rendered_items": rendered,
        "abstention_count": len(abstentions),
        "abstentions": abstentions,
        "manual_outputs_reviewed": len(visuals),
        "manually_usable_outputs": len(accepted),
        "rendered_usable_rate": rendered_rate,
        "cohort_salvage_rate": cohort_rate,
        "failure_counts": dict(sorted(failures.items())),
        "gate_checks": checks,
        "candidate_for_fresh_source_disjoint_holdout": promoted,
        "automatic_keep_rule_promoted": False,
        "corpus_mutated": False,
        "policy": "shadow_only_failed_mechanisms_abstain",
    }
    return report, accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post-reveal", type=Path, required=True)
    parser.add_argument("--materialization", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.accepted.exists():
        raise SystemExit("refusing to overwrite evaluation artifacts")
    report, accepted = evaluate(
        read_jsonl(args.manifest), read_tsv(args.blind), read_tsv(args.post_reveal),
        json.loads(args.materialization.read_text()), json.loads(args.contract.read_text()),
    )
    args.accepted.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted))
    report["artifact_sha256"] = {
        "manifest": sha256(args.manifest),
        "blind": sha256(args.blind),
        "post_reveal": sha256(args.post_reveal),
        "materialization": sha256(args.materialization),
        "contract": sha256(args.contract),
        "accepted": sha256(args.accepted),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["candidate_for_fresh_source_disjoint_holdout"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
