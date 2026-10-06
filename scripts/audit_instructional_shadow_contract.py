#!/usr/bin/env python3
"""Replay the deterministic instructional shadow contract against a complete manual audit."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

try:
    from scripts.run_instructional_shadow_compare import shadow_contract_rejection_reasons
except ModuleNotFoundError:  # Support direct execution as scripts/<name>.py.
    from run_instructional_shadow_compare import shadow_contract_rejection_reasons


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("manual_review", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    manifest = load_json(args.manifest)
    with args.manual_review.open(newline="") as handle:
        review_rows = list(csv.DictReader(handle, delimiter="\t"))
    reviews = {int(row["ordinal"]): row for row in review_rows}
    items = manifest.get("items") or []
    ordinals = {int(item["ordinal"]) for item in items}
    if set(reviews) != ordinals:
        missing = sorted(ordinals - set(reviews))
        extra = sorted(set(reviews) - ordinals)
        raise SystemExit(f"manual review must cover every output exactly; missing={missing}, extra={extra}")

    results = []
    for item in items:
        ordinal = int(item["ordinal"])
        reasons = shadow_contract_rejection_reasons(item["demo"])
        results.append(
            {
                "ordinal": ordinal,
                "uid": item["uid"],
                "shadow_index": item["shadow_index"],
                "manual_decision": reviews[ordinal]["decision"],
                "contract_pass": not reasons,
                "contract_rejection_reasons": reasons,
            }
        )

    filtered = [row for row in results if not row["contract_pass"]]
    survivors = [row for row in results if row["contract_pass"]]
    accepted_survivors = [row for row in survivors if row["manual_decision"] == "accept"]
    definite_false_positive_survivors = [
        row for row in survivors if row["manual_decision"] == "reject"
    ]
    unresolved_survivors = [
        row for row in survivors if row["manual_decision"] == "uncertain"
    ]
    false_rejections = [row for row in filtered if row["manual_decision"] == "accept"]
    payload = {
        "kind": "instructional_shadow_contract_manual_replay",
        "source_manifest": str(args.manifest),
        "manual_review": str(args.manual_review),
        "outputs_reviewed": len(results),
        "audit_coverage": 1.0,
        "filtered_outputs": len(filtered),
        "filtered_ordinals": [row["ordinal"] for row in filtered],
        "filtered_manual_rejects": sum(row["manual_decision"] == "reject" for row in filtered),
        "false_rejection_ordinals": [row["ordinal"] for row in false_rejections],
        "surviving_outputs": len(survivors),
        "accepted_survivor_ordinals": [row["ordinal"] for row in accepted_survivors],
        "definite_false_positive_survivor_ordinals": [
            row["ordinal"] for row in definite_false_positive_survivors
        ],
        "unresolved_survivor_ordinals": [row["ordinal"] for row in unresolved_survivors],
        "accepted_precision_among_survivors": (
            len(accepted_survivors) / len(survivors) if survivors else None
        ),
        "production_eligible": bool(survivors)
        and not false_rejections
        and len(accepted_survivors) == len(survivors),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in (
        "outputs_reviewed", "filtered_outputs", "surviving_outputs",
        "accepted_precision_among_survivors", "production_eligible"
    )}))


if __name__ == "__main__":
    main()
