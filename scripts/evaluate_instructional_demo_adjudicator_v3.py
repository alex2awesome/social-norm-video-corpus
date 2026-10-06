#!/usr/bin/env python3
"""Evaluate V3 on a manually audited instructional cohort."""

from __future__ import annotations

import argparse
import csv
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_instructional_demo_consensus_v2 import latest_success, metric, read_jsonl
except ModuleNotFoundError:
    from evaluate_instructional_demo_consensus_v2 import latest_success, metric, read_jsonl  # type: ignore[no-redef]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_index(row: dict[str, Any]) -> int:
    """Return the stable audit index used by both old and new manifests."""
    value = row.get("audit_index", row.get("storyboard_index"))
    if value is None:
        raise ValueError("manifest row has no audit_index or storyboard_index")
    return int(value)


def binary_label(row: dict[str, str]) -> bool:
    """Parse either generation of the manual visual-demo ledger."""
    value = row.get("visual_demo", row.get("manual_visual_demo", ""))
    normalized = value.strip().lower()
    if normalized in {"y", "yes", "true", "1"}:
        return True
    if normalized in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"invalid manual visual-demo label: {value!r}")


def evaluate(
    manifest: list[dict[str, Any]],
    manual: list[dict[str, str]],
    outputs: list[dict[str, Any]],
    reviews: list[dict[str, str]],
    *,
    kind: str = "instructional_demo_adjudicator_v3_development_evaluation",
    status: str = "development_only_not_eligible_for_operational_registration",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_index = {audit_index(row): row for row in manifest}
    gold = {int(row["audit_index"]): row for row in manual}
    review = {int(row["audit_index"]): row for row in reviews}
    model = latest_success(outputs)
    # The manual ledger may include render failures that correctly do not
    # appear in the model manifest.  Every scored model item must still have
    # gold, while output reviews must match scored coverage exactly.
    if not set(by_index).issubset(set(gold)) or set(by_index) != set(review):
        raise ValueError("manifest/manual/review coverage mismatch")
    if {str(row["item_id"]) for row in manifest} != set(model):
        raise ValueError("manifest/model successful coverage mismatch")
    review_complete = bool(review) and all(
        row.get("output_reviewed", "").strip().lower()
        in {"y", "yes", "true", "1"}
        for row in review.values()
    )
    rows = []
    for index in sorted(by_index):
        source = by_index[index]
        item_id = str(source["item_id"])
        label = binary_label(gold[index])
        prediction = bool(model[item_id]["result"]["demo_candidate"])
        rows.append({
            "audit_index": index,
            "item_id": item_id,
            "manual_visual_demo": label,
            "v3_demo_candidate": prediction,
            "error": (
                "false_positive" if prediction and not label
                else "false_negative" if not prediction and label
                else None
            ),
            "result": model[item_id]["result"],
            "manual_output_reviewed": (
                review[index].get("output_reviewed", "").strip().lower()
                in {"y", "yes", "true", "1"}
            ),
            "manual_output_review_note": review[index].get("note"),
        })
    truth = [row["manual_visual_demo"] for row in rows]
    result = metric(truth, [row["v3_demo_candidate"] for row in rows])
    support_values = [
        review[index].get("output_visually_supported", "").strip().lower()
        for index in sorted(by_index)
    ]
    if any(value not in {"yes", "no"} for value in support_values):
        raise ValueError("manual output review lacks visual-support adjudication")
    mechanisms = collections.Counter(
        review[index].get("error_mechanism", "").strip() or "unspecified"
        for index in sorted(by_index)
        if review[index].get("error_mechanism", "").strip() not in {"", "none"}
    )
    return rows, {
        "kind": kind,
        "items": len(rows),
        "manual_visual_demos": sum(truth),
        "metric": result,
        "manual_output_review_complete": review_complete,
        "manual_outputs_reviewed": len(review),
        "model_outputs_visually_supported": support_values.count("yes"),
        "model_output_visual_support_rate": support_values.count("yes") / len(rows),
        "manual_error_mechanisms": dict(sorted(mechanisms.items())),
        "status": status,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
        "all_media_retained": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manual-ledger", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--manual-output-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument(
        "--evaluation-kind",
        default="instructional_demo_adjudicator_v3_development_evaluation",
    )
    parser.add_argument(
        "--status",
        default="development_only_not_eligible_for_operational_registration",
    )
    args = parser.parse_args()
    rows, summary = evaluate(
        read_jsonl(args.manifest), read_tsv(args.manual_ledger),
        read_jsonl(args.outputs), read_tsv(args.manual_output_review),
        kind=args.evaluation_kind, status=args.status,
    )
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary["artifact_sha256"] = {
        "manifest": sha256(args.manifest),
        "manual_ledger": sha256(args.manual_ledger),
        "outputs": sha256(args.outputs),
        "manual_output_review": sha256(args.manual_output_review),
        "evaluated": sha256(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
