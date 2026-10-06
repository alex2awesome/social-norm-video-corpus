#!/usr/bin/env python3
"""Evaluate every transformed commentary candidate after two-pass review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


TRINARY = {"yes", "no", "uncertain"}
MEDIA = {"live_action", "animation", "dramatized", "mixed", "other", "uncertain"}
ALIGNMENTS = {"exact", "same_norm", "mismatch", "not_usable", "uncertain"}
ROUTES = {
    "commentary_visual", "instructional_demo", "relabel_required", "reject",
    "uncertain",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def keyed(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get("candidate_id") or ""): row for row in rows}
    if "" in output or len(output) != len(rows):
        raise ValueError(f"{name} has empty or duplicate candidate_id")
    return output


def validate(
    manifest: list[dict[str, Any]],
    blind: list[dict[str, str]],
    post: list[dict[str, str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    manifests = keyed(manifest, "manifest")
    blinds = keyed(blind, "blind ledger")
    posts = keyed(post, "post-reveal ledger")
    if not (set(manifests) == set(blinds) == set(posts)):
        raise ValueError("manual ledgers do not exactly cover transformed manifest")
    blind_trinary = (
        "performed_event_visible", "actor_target_grounded", "start_boundary_clean",
        "end_boundary_clean", "label_bearing_text_absent", "audio_absent",
    )
    for candidate_id in manifests:
        for field in blind_trinary:
            if blinds[candidate_id].get(field) not in TRINARY:
                raise ValueError(f"{candidate_id}: invalid or missing {field}")
        if blinds[candidate_id].get("medium") not in MEDIA:
            raise ValueError(f"{candidate_id}: invalid or missing medium")
        if not blinds[candidate_id].get("blind_evidence", "").strip():
            raise ValueError(f"{candidate_id}: missing blind_evidence")
        if posts[candidate_id].get("exact_named_action_visible") not in TRINARY:
            raise ValueError(f"{candidate_id}: invalid exact_named_action_visible")
        if posts[candidate_id].get("label_alignment") not in ALIGNMENTS:
            raise ValueError(f"{candidate_id}: invalid label_alignment")
        if posts[candidate_id].get("route") not in ROUTES:
            raise ValueError(f"{candidate_id}: invalid route")
        if not posts[candidate_id].get("post_reveal_evidence", "").strip():
            raise ValueError(f"{candidate_id}: missing post_reveal_evidence")
    return manifests, blinds, posts


def evaluate(
    manifest: list[dict[str, Any]], blind: list[dict[str, str]],
    post: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifests, blinds, posts = validate(manifest, blind, post)
    accepted: list[dict[str, Any]] = []
    failure_counts: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    for candidate_id, media in manifests.items():
        visual = blinds[candidate_id]
        semantic = posts[candidate_id]
        route = semantic["route"]
        route_counts[route] = route_counts.get(route, 0) + 1
        checks = {
            "performed_event_visible": visual["performed_event_visible"] == "yes",
            "actor_target_grounded": visual["actor_target_grounded"] == "yes",
            "start_boundary_clean": visual["start_boundary_clean"] == "yes",
            "end_boundary_clean": visual["end_boundary_clean"] == "yes",
            "label_bearing_text_absent": visual["label_bearing_text_absent"] == "yes",
            "audio_absent": visual["audio_absent"] == "yes",
            "exact_named_action_visible": semantic["exact_named_action_visible"] == "yes",
            "label_alignment_exact": semantic["label_alignment"] == "exact",
            "commentary_visual_route": route == "commentary_visual",
        }
        for name, passed in checks.items():
            if not passed:
                failure_counts[name] = failure_counts.get(name, 0) + 1
        if all(checks.values()):
            accepted.append({
                "candidate_id": candidate_id,
                "audit_index": int(media["audit_index"]),
                "proxy_clip": media["proxy_clip"],
                "proxy_clip_sha256": media["proxy_clip_sha256"],
                "manual_acceptance": "exact_commentary_visual_clip",
                "automatic_acceptance": False,
                "corpus_disposition": None,
                "policy": "individual_manual_acceptance_preserve_source",
            })
    report = {
        "kind": "commentary_clip_plan_manual_evaluation_v1",
        "items": len(manifests),
        "manual_review_complete": True,
        "exact_commentary_visual_clips": len(accepted),
        "exact_commentary_visual_rate": len(accepted) / len(manifests),
        "route_counts": dict(sorted(route_counts.items())),
        "failed_check_counts": dict(sorted(failure_counts.items())),
        "automatic_acceptance_rule_promoted": False,
        "corpus_mutated": False,
        "policy": "individual_manual_acceptance_only_no_automatic_rule",
    }
    return report, accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post-reveal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.accepted.exists():
        raise SystemExit("refusing to overwrite evaluation artifacts")
    report, accepted = evaluate(
        read_jsonl(args.manifest), read_tsv(args.blind), read_tsv(args.post_reveal)
    )
    args.accepted.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted))
    report["artifact_sha256"] = {
        "manifest": sha256(args.manifest),
        "blind": sha256(args.blind),
        "post_reveal": sha256(args.post_reveal),
        "accepted": sha256(args.accepted),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
