#!/usr/bin/env python3
"""Validate final manual judgments for exact audio-policy artifact candidates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.materialize_post_localization_candidates_v1 import REVIEW_FIELDS, sha256
except ModuleNotFoundError:
    from materialize_post_localization_candidates_v1 import REVIEW_FIELDS, sha256  # type: ignore[no-redef]


YES_NO = {"yes", "no"}
GATE_FIELDS = (
    "visual_event_complete", "actor_target_grounded",
    "exact_behavior_label_supported", "start_boundary_clean",
    "end_boundary_clean", "label_bearing_visible_text_absent",
    "audio_policy_correct", "label_bearing_audio_absent",
    "required_behavior_audio_preserved", "artifact_integrity",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate(manifest: list[dict[str, Any]], reviews: list[dict[str, str]]) -> dict[str, Any]:
    expected = {str(row.get("candidate_id") or ""): row for row in manifest}
    observed = {str(row.get("candidate_id") or ""): row for row in reviews}
    if not expected or not all(expected) or len(expected) != len(manifest):
        raise ValueError("artifact manifest has missing or duplicate candidate ids")
    if set(observed) != set(expected) or len(observed) != len(reviews):
        raise ValueError("manual review does not exactly cover artifact manifest")
    accepted = []
    for candidate_id, row in observed.items():
        if set(row) != set(REVIEW_FIELDS):
            raise ValueError(f"{candidate_id}: post-transform review schema mismatch")
        lineage = expected[candidate_id]
        if row.get("pillar") != str(lineage["pillar"]) or row.get("uid") != str(lineage["uid"]):
            raise ValueError(f"{candidate_id}: lineage mismatch")
        if lineage.get("approval_status") != "awaiting_post_transform_manual_audit":
            raise ValueError(f"{candidate_id}: artifact was not awaiting manual audit")
        artifact_path = Path(str(lineage.get("artifact_path") or ""))
        if (
            not artifact_path.is_file()
            or sha256(artifact_path) != lineage.get("artifact_sha256")
        ):
            raise ValueError(f"{candidate_id}: artifact missing or hash-mismatched")
        for field in GATE_FIELDS:
            if row.get(field) not in YES_NO:
                raise ValueError(f"{candidate_id}: missing {field}")
        if not row.get("manual_description", "").strip() or not row.get("manual_rationale", "").strip():
            raise ValueError(f"{candidate_id}: missing manual evidence")
        expected_accept = all(row[field] == "yes" for field in GATE_FIELDS)
        if row.get("final_manual_accept") not in YES_NO:
            raise ValueError(f"{candidate_id}: missing final_manual_accept")
        if (row["final_manual_accept"] == "yes") != expected_accept:
            raise ValueError(f"{candidate_id}: final acceptance contradicts atomic fields")
        if expected_accept:
            accepted.append({
                "candidate_id": candidate_id,
                "pillar": row["pillar"],
                "uid": row["uid"],
                "artifact_path": lineage["artifact_path"],
                "artifact_sha256": lineage["artifact_sha256"],
                "behavior_label": lineage["behavior_label"],
                "acceptance_basis": "complete_manual_exact_hashed_artifact_review_v1",
                "source_media_preserved": True,
            })
    return {
        "kind": "post_transform_exact_artifact_manual_evaluation_v1",
        "reviewed": len(reviews),
        "manual_review_coverage": 1.0,
        "accepted": len(accepted),
        "accepted_artifacts": accepted,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
        "source_media_preserved": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = validate(read_jsonl(args.manifest), read_tsv(args.review))
    report["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    report["review_sha256"] = hashlib.sha256(args.review.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
