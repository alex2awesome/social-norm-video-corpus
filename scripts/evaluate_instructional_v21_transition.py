#!/usr/bin/env python3
"""Audit every V21 transition-rule output against frozen manual judgments."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_success(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("error") in (None, "") and isinstance(row.get("result"), dict):
            output[str(row["item_id"])] = row
    return output


def target_metric(
    rows: list[dict[str, Any]],
    target: str,
) -> dict[str, Any]:
    accepted = [row for row in rows if row["v21_accept"]]
    positives = [row for row in rows if row[target]]
    true_positive = [row for row in accepted if row[target]]
    return {
        "accepted": len(accepted),
        "positive": len(true_positive),
        "negative": len(accepted) - len(true_positive),
        "precision": len(true_positive) / len(accepted) if accepted else None,
        "recall": len(true_positive) / len(positives) if positives else None,
        "accepted_indices": [row["audit_index"] for row in accepted],
        "false_positive_indices": [
            row["audit_index"] for row in accepted if not row[target]
        ],
        "false_negative_indices": [
            row["audit_index"]
            for row in rows
            if not row["v21_accept"] and row[target]
        ],
    }


def evaluate(
    manual_rows: list[dict[str, Any]],
    vlm_rows: list[dict[str, Any]],
    *,
    minimum_precision: float = 0.9,
    minimum_support: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(manual_rows) != 60:
        raise ValueError("expected 60 frozen manual rows")
    by_item = latest_success(vlm_rows)
    output: list[dict[str, Any]] = []
    for manual in manual_rows:
        item_id = str(manual["item_id"])
        record = by_item.get(item_id)
        if record is None:
            raise ValueError(f"missing successful V21 record: {item_id}")
        result = record["result"]
        output.append(
            {
                **manual,
                "v21_model": record["model"],
                "v21_rubric": record["rubric"],
                "v21_accept": result.get("demo_usable") == "yes",
                "v21_demo_usable": result.get("demo_usable"),
                "v21_scene_role": result.get("scene_role"),
                "v21_social_scope": result.get("social_scope"),
                "v21_rejection_reason": result.get("rejection_reason"),
                "v21_action": result.get(
                    "literal_action_or_situated_utterance"
                ),
                "v21_evidence": result.get("evidence"),
            }
        )
    if len(by_item) != 60:
        extra = sorted(set(by_item) - {str(row["item_id"]) for row in manual_rows})
        if extra:
            raise ValueError(f"unexpected V21 records: {extra[:5]}")

    metrics = {
        "visual": target_metric(output, "manual_visual_demo"),
        "relabel_usable": target_metric(output, "manual_relabel_usable"),
        "exact_original": target_metric(output, "manual_exact_original"),
    }
    passed = (
        metrics["visual"]["accepted"] >= minimum_support
        and metrics["visual"]["precision"] is not None
        and metrics["visual"]["precision"] >= minimum_precision
        and metrics["relabel_usable"]["precision"] is not None
        and metrics["relabel_usable"]["precision"] >= minimum_precision
    )
    accepted = [row for row in output if row["v21_accept"]]
    false_positives = [
        {
            "audit_index": row["audit_index"],
            "candidate_id": row["candidate_id"],
            "visual_form": row["visual_form"],
            "manual_note": row["blind_visual_note"],
            "v21_scene_role": row["v21_scene_role"],
            "v21_action": row["v21_action"],
            "v21_evidence": row["v21_evidence"],
        }
        for row in accepted
        if not row["manual_visual_demo"]
    ]
    false_negatives = [
        {
            "audit_index": row["audit_index"],
            "candidate_id": row["candidate_id"],
            "visual_form": row["visual_form"],
            "manual_note": row["blind_visual_note"],
            "v21_rejection_reason": row["v21_rejection_reason"],
            "v21_evidence": row["v21_evidence"],
        }
        for row in output
        if not row["v21_accept"] and row["manual_visual_demo"]
    ]
    summary = {
        "kind": "instructional_v21_event_transition_manual_output_audit",
        "status": "pass_shadow_only" if passed else "failed_no_promotion",
        "policy": "read_only_no_keep_reject_or_corpus_mutation",
        "gate": {
            "minimum_visual_precision": minimum_precision,
            "minimum_relabel_usable_precision": minimum_precision,
            "minimum_accepted_support": minimum_support,
            "passed": passed,
        },
        "coverage": {
            "manual_rows": len(manual_rows),
            "successful_vlm_rows": len(by_item),
            "all_outputs_manually_scored": len(output),
        },
        "decision_counts": dict(
            Counter(str(row["v21_demo_usable"]) for row in output)
        ),
        "metrics": metrics,
        "visual_false_positives": false_positives,
        "visual_false_negatives": false_negatives,
    }
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--vlm", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.9)
    parser.add_argument("--minimum-support", type=int, default=5)
    args = parser.parse_args()
    rows, summary = evaluate(
        read_jsonl(args.manual),
        read_jsonl(args.vlm),
        minimum_precision=args.minimum_precision,
        minimum_support=args.minimum_support,
    )
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary["artifact_sha256"] = {
        "manual": sha256(args.manual),
        "vlm": sha256(args.vlm),
        "evaluated": sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
