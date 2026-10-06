#!/usr/bin/env python3
"""Evaluate the post-hoc V15 utterance-anchor hypothesis on V14 labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable

try:
    from scripts.evaluate_instructional_v11_holdout import metric, read_jsonl, sha256
    from scripts.score_instructional_v15_utterance_anchor import utterance_anchor
except ModuleNotFoundError:
    from evaluate_instructional_v11_holdout import metric, read_jsonl, sha256
    from score_instructional_v15_utterance_anchor import utterance_anchor


def keyed(rows: list[dict[str, Any]], name: str) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = int(row["audit_index"])
        if index in output:
            raise ValueError(f"{name}: duplicate audit_index {index}")
        output[index] = row
    return output


def score(
    selection_path: Path,
    evaluated_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selections = keyed(read_jsonl(selection_path), "selection")
    evaluated = keyed(read_jsonl(evaluated_path), "evaluated")
    if set(selections) != set(evaluated):
        raise ValueError("selection/evaluated audit-index coverage mismatch")

    rows: list[dict[str, Any]] = []
    for index in sorted(selections):
        source = selections[index]
        manual = evaluated[index]
        if source["item_id"] != manual["item_id"]:
            raise ValueError(f"audit_index {index}: item_id mismatch")
        if source["band"] != "primary_v14_candidate":
            continue
        anchor = utterance_anchor(source["v9a"])
        rows.append(
            {
                "audit_index": index,
                "candidate_id": source["candidate_id"],
                "item_id": source["item_id"],
                "uid": source["uid"],
                "anchor_pass": anchor["passes"],
                "anchor_reason": anchor["reason"],
                "anchor_evidence_field": anchor["evidence_field"],
                "anchor_quoted_utterance": anchor["quoted_utterance"],
                "manual_visual_demo": bool(manual["manual_visual_demo"]),
                "manual_usable_after_relabel": bool(
                    manual["manual_usable_after_relabel"]
                ),
                "manual_exact_original": bool(manual["manual_exact_original"]),
                "manual_failure_mechanism": manual["manual_failure_mechanism"],
            }
        )

    retained = [row for row in rows if row["anchor_pass"]]
    rejected = [row for row in rows if not row["anchor_pass"]]

    def summarize(
        subset: list[dict[str, Any]],
        predicate: Callable[[dict[str, Any]], bool],
    ) -> dict[str, Any]:
        return metric(subset, predicate)

    summary = {
        "kind": "instructional_v15_utterance_anchor_posthoc_development",
        "status": "POSTHOC DEVELOPMENT ONLY; NOT VALIDATION OR PROMOTION.",
        "rule": {
            "base": (
                "V14 candidate: V9A strict visual + V10A demo_usable=yes + "
                "V10A social_response_or_consequence_present=yes"
            ),
            "addition": (
                "When V9A evidence_source=situated_dialogue_or_subtitles, "
                "require a quoted utterance with >=2 word tokens in one of "
                "the five frozen V9A event-evidence fields. Physical/static "
                "actions pass unchanged."
            ),
        },
        "coverage": {
            "v14_candidates": len(rows),
            "retained": len(retained),
            "rejected": len(rejected),
            "retained_uids": len({row["uid"] for row in retained}),
        },
        "retained_metrics": {
            "visual_demo": summarize(
                retained, lambda row: row["manual_visual_demo"]
            ),
            "usable_after_relabel": summarize(
                retained, lambda row: row["manual_usable_after_relabel"]
            ),
            "exact_original": summarize(
                retained, lambda row: row["manual_exact_original"]
            ),
        },
        "rejected_metrics": {
            "visual_demo": summarize(
                rejected, lambda row: row["manual_visual_demo"]
            ),
            "usable_after_relabel": summarize(
                rejected, lambda row: row["manual_usable_after_relabel"]
            ),
            "exact_original": summarize(
                rejected, lambda row: row["manual_exact_original"]
            ),
        },
        "retained_visual_false_positive_indices": [
            row["audit_index"] for row in retained if not row["manual_visual_demo"]
        ],
        "rejected_visual_true_positive_indices": [
            row["audit_index"] for row in rejected if row["manual_visual_demo"]
        ],
        "rejected_indices": [row["audit_index"] for row in rejected],
        "retained_exact_failure_mechanisms": dict(
            sorted(
                Counter(
                    row["manual_failure_mechanism"]
                    for row in retained
                    if not row["manual_exact_original"]
                ).items()
            )
        ),
        "promotion_eligible": False,
        "corpus_mutated": False,
        "next_step": (
            "Freeze this rule before collecting and blindly auditing a new "
            "source-disjoint YouTube violation cohort."
        ),
        "artifact_sha256": {
            "selection": sha256(selection_path),
            "evaluated_v14": sha256(evaluated_path),
        },
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--evaluated-v14", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.output, args.summary_output):
        if path.exists():
            raise SystemExit(f"refusing to overwrite: {path}")
    rows, summary = score(args.selection, args.evaluated_v14)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary["artifact_sha256"]["output"] = sha256(args.output)
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

