#!/usr/bin/env python3
"""Validate lineage from witnessed transcript proposals to rendered VLM windows.

The proposal inventory includes explicit negative controls that the frozen
renderer deliberately does not materialize.  This validator accounts for those
rows explicitly, verifies that every eligible proposal has exactly one rendered
manifest row, and never changes corpus state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proposal_id(parent: dict[str, Any], candidate: dict[str, Any]) -> str:
    return f"{parent['item_id']}:candidate_{int(candidate['segment_index'])}"


def summarize(
    proposal_rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    *,
    expected_proposals: int | None = None,
    expected_proposal_parents: int | None = None,
    expected_render_source_items: int | None = None,
) -> dict[str, Any]:
    eligible: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    exclusion_counts: Counter[str] = Counter()
    proposed = 0
    duplicate_eligible: list[str] = []
    for parent in proposal_rows:
        for candidate in parent.get("candidates") or []:
            proposed += 1
            self_defense = bool(candidate.get("negative_self_defense"))
            reported = bool(candidate.get("negative_reported"))
            if self_defense or reported:
                if self_defense:
                    exclusion_counts["negative_self_defense"] += 1
                if reported:
                    exclusion_counts["negative_reported"] += 1
                exclusion_counts[
                    "both" if self_defense and reported else "excluded_union"
                ] += 1
                continue
            candidate_id = proposal_id(parent, candidate)
            if candidate_id in eligible:
                duplicate_eligible.append(candidate_id)
            eligible[candidate_id] = (parent, candidate)

    manifest_by_id: dict[str, dict[str, Any]] = {}
    duplicate_manifest: list[str] = []
    for row in manifest_rows:
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in manifest_by_id:
            duplicate_manifest.append(candidate_id)
        manifest_by_id[candidate_id] = row

    eligible_ids = set(eligible)
    manifest_ids = set(manifest_by_id)
    render_source_items = {
        str(row.get("source_item_id") or row.get("item_id") or "")
        for row in manifest_rows
        if row.get("source_item_id") or row.get("item_id")
    }
    missing = sorted(eligible_ids - manifest_ids)
    unexpected = sorted(manifest_ids - eligible_ids)
    field_mismatches: list[str] = []
    for candidate_id in sorted(eligible_ids & manifest_ids):
        parent, candidate = eligible[candidate_id]
        rendered = manifest_by_id[candidate_id]
        expected_values = {
            "item_id": str(parent.get("item_id") or ""),
            "uid": str(parent.get("uid") or ""),
            "candidate_text": str(candidate.get("text") or ""),
        }
        if any(str(rendered.get(key) or "") != value for key, value in expected_values.items()):
            field_mismatches.append(candidate_id)

    issues: list[dict[str, Any]] = []
    checks = (
        ("proposal_count_mismatch", expected_proposals is not None and proposed != expected_proposals,
         {"expected": expected_proposals, "actual": proposed}),
        ("proposal_parent_count_mismatch", expected_proposal_parents is not None and len(proposal_rows) != expected_proposal_parents,
         {"expected": expected_proposal_parents, "actual": len(proposal_rows)}),
        ("render_source_item_count_mismatch", expected_render_source_items is not None and len(render_source_items) != expected_render_source_items,
         {"expected": expected_render_source_items, "actual": len(render_source_items)}),
        ("duplicate_eligible_candidate_id", bool(duplicate_eligible), duplicate_eligible[:20]),
        ("duplicate_manifest_candidate_id", bool(duplicate_manifest), duplicate_manifest[:20]),
        ("eligible_candidate_missing_from_manifest", bool(missing), missing[:20]),
        ("unexpected_manifest_candidate", bool(unexpected), unexpected[:20]),
        ("proposal_manifest_field_mismatch", bool(field_mismatches), field_mismatches[:20]),
    )
    for kind, failed, detail in checks:
        if failed:
            issues.append({"kind": kind, "detail": detail})

    excluded_union = proposed - len(eligible_ids)
    return {
        "kind": "witnessed_video_asr_population_lineage_v1",
        "proposal_parent_rows": len(proposal_rows),
        "unique_render_source_items": len(render_source_items),
        "proposed_candidate_windows": proposed,
        "excluded_candidate_windows": excluded_union,
        "exclusion_flag_counts": dict(sorted(exclusion_counts.items())),
        "eligible_candidate_windows": len(eligible_ids),
        "rendered_manifest_rows": len(manifest_rows),
        "unique_rendered_candidate_ids": len(manifest_ids),
        "coverage_fraction": (
            len(eligible_ids & manifest_ids) / len(eligible_ids) if eligible_ids else None
        ),
        "issues": issues,
        "passed": not issues,
        "policy": "validation_only_no_keep_reject_or_corpus_mutation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-proposals", type=int)
    parser.add_argument("--expected-proposal-parents", type=int)
    parser.add_argument("--expected-render-source-items", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    report = summarize(
        read_jsonl(args.proposals),
        read_jsonl(args.manifest),
        expected_proposals=args.expected_proposals,
        expected_proposal_parents=args.expected_proposal_parents,
        expected_render_source_items=args.expected_render_source_items,
    )
    report["artifact_sha256"] = {
        "proposals": sha256(args.proposals),
        "manifest": sha256(args.manifest),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
