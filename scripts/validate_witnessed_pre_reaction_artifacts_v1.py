#!/usr/bin/env python3
"""Validate exhaustive manual audits of exact pre-reaction artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.materialize_witnessed_pre_reaction_splices_v1 import REVIEW_FIELDS, sha256
except ModuleNotFoundError:
    from materialize_witnessed_pre_reaction_splices_v1 import REVIEW_FIELDS, sha256  # type: ignore[no-redef]


GATES = (
    "social_action_complete", "exact_behavior_label_supported",
    "reaction_visible_absent", "reaction_audible_absent",
    "start_boundary_clean", "end_boundary_clean", "artifact_integrity",
)


def validate(manifest: list[dict[str, Any]], rows: list[dict[str, str]]) -> dict[str, Any]:
    expected = {str(row["candidate_id"]): row for row in manifest}
    reviews = {str(row.get("candidate_id") or ""): row for row in rows}
    if not expected or len(expected) != len(manifest) or set(expected) != set(reviews) or len(reviews) != len(rows):
        raise ValueError("post-splice review does not exactly cover manifest")
    accepted = []
    for candidate_id, row in reviews.items():
        lineage = expected[candidate_id]
        if set(row) != set(REVIEW_FIELDS) or row.get("uid") != str(lineage["uid"]):
            raise ValueError(f"{candidate_id}: schema or lineage mismatch")
        if lineage.get("approval_status") != "awaiting_post_splice_manual_audit":
            raise ValueError(f"{candidate_id}: artifact was not awaiting manual audit")
        artifact_path = Path(str(lineage.get("artifact_path") or ""))
        if (
            not artifact_path.is_file()
            or sha256(artifact_path) != lineage.get("artifact_sha256")
        ):
            raise ValueError(f"{candidate_id}: artifact missing or hash-mismatched")
        if any(row.get(field) not in {"yes", "no"} for field in GATES):
            raise ValueError(f"{candidate_id}: incomplete atomic audit")
        if not row.get("manual_description", "").strip() or not row.get("manual_rationale", "").strip():
            raise ValueError(f"{candidate_id}: missing manual evidence")
        passed = all(row[field] == "yes" for field in GATES)
        if row.get("final_manual_accept") not in {"yes", "no"} or (
            row["final_manual_accept"] == "yes"
        ) != passed:
            raise ValueError(f"{candidate_id}: inconsistent final_manual_accept")
        if passed:
            accepted.append({
                "candidate_id": candidate_id,
                "uid": row["uid"],
                "artifact_path": lineage["artifact_path"],
                "artifact_sha256": lineage["artifact_sha256"],
                "behavior_label": lineage["behavior_label"],
                "acceptance_basis": "manual_exact_pre_reaction_artifact_audit_v1",
            })
    return {
        "kind": "witnessed_pre_reaction_exact_artifact_evaluation_v1",
        "reviewed": len(rows), "accepted": len(accepted),
        "accepted_artifacts": accepted,
        "reaction_audio_absence_explicitly_audited": True,
        "automatic_acceptance": False, "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    with args.review.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    report = validate(manifest, rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
