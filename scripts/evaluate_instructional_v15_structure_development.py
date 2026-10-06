#!/usr/bin/env python3
"""Compare the post-hoc Gemma V15 structure rubric to frozen V14 humans."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any, Callable

try:
    from scripts.evaluate_instructional_v11_holdout import (
        indexed,
        metric,
        read_jsonl,
        sha256,
    )
except ModuleNotFoundError:
    from evaluate_instructional_v11_holdout import (  # type: ignore[no-redef]
        indexed,
        metric,
        read_jsonl,
        sha256,
    )


def confusion(
    rows: list[dict[str, Any]],
    target: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    tp = sum(row["gemma_pass"] and target(row) for row in rows)
    fp = sum(row["gemma_pass"] and not target(row) for row in rows)
    fn = sum(not row["gemma_pass"] and target(row) for row in rows)
    tn = sum(not row["gemma_pass"] and not target(row) for row in rows)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def evaluate(
    gemma_path: Path,
    manual_path: Path,
) -> dict[str, Any]:
    manual_rows = read_jsonl(manual_path)
    expected = set(range(len(manual_rows)))
    manual = indexed(manual_rows, "manual", expected)
    gemma = indexed(read_jsonl(gemma_path), "gemma", expected)
    rows: list[dict[str, Any]] = []
    for audit_index in sorted(expected):
        human = manual[audit_index]
        model = gemma[audit_index]
        if model.get("candidate_id") != human.get("candidate_id"):
            raise ValueError(f"audit_index {audit_index}: identity mismatch")
        if model.get("error") is not None or model.get("result") is None:
            raise ValueError(f"audit_index {audit_index}: incomplete Gemma result")
        prediction = model["result"].get("structural_demo_pass")
        if prediction not in {"yes", "no", "uncertain"}:
            raise ValueError(f"audit_index {audit_index}: invalid model output")
        rows.append(
            {
                "audit_index": audit_index,
                "is_candidate": bool(human["is_v14_candidate"]),
                "gemma_pass": prediction == "yes",
                "gemma_raw": prediction,
                "manual_visual_demo": bool(human["manual_visual_demo"]),
                "manual_usable_after_relabel": bool(
                    human["manual_usable_after_relabel"]
                ),
                "manual_exact_original": bool(
                    human["manual_exact_original"]
                ),
                "manual_failure_mechanism": human[
                    "manual_failure_mechanism"
                ],
                "model_structure": model["result"][
                    "interaction_structure"
                ],
                "model_evidence": model["result"]["evidence"],
            }
        )

    candidates = [row for row in rows if row["is_candidate"]]
    conjunction = [row for row in candidates if row["gemma_pass"]]
    human_visual = lambda row: row["manual_visual_demo"]
    human_relabel = lambda row: row["manual_usable_after_relabel"]
    human_exact = lambda row: row["manual_exact_original"]
    model_false_positives = [
        row for row in rows if row["gemma_pass"] and not human_visual(row)
    ]
    return {
        "kind": "instructional_v15_structure_posthoc_development",
        "development_only": True,
        "source_disjoint_validation": False,
        "coverage": {
            "rows": len(rows),
            "candidate_rows": len(candidates),
            "gemma_pass": sum(row["gemma_pass"] for row in rows),
            "gemma_fail_or_uncertain": sum(
                not row["gemma_pass"] for row in rows
            ),
        },
        "gemma_vs_manual": {
            "visual_demo": confusion(rows, human_visual),
            "usable_after_relabel": confusion(rows, human_relabel),
            "exact_original": confusion(rows, human_exact),
        },
        "v14_candidate_and_gemma_pass": {
            "selected": len(conjunction),
            "visual_demo": metric(conjunction, human_visual),
            "usable_after_relabel": metric(conjunction, human_relabel),
            "exact_original": metric(conjunction, human_exact),
        },
        "candidate_false_positive_retention": {
            "v14_candidate_visual_false_positives": [
                row["audit_index"]
                for row in candidates
                if not human_visual(row)
            ],
            "retained_by_gemma": [
                row["audit_index"]
                for row in candidates
                if row["gemma_pass"] and not human_visual(row)
            ],
        },
        "gemma_visual_false_positive_indices": [
            row["audit_index"] for row in model_false_positives
        ],
        "gemma_visual_false_positive_mechanisms": dict(
            sorted(
                Counter(
                    row["manual_failure_mechanism"]
                    for row in model_false_positives
                ).items()
            )
        ),
        "gemma_visual_false_negative_indices": [
            row["audit_index"]
            for row in rows
            if not row["gemma_pass"] and human_visual(row)
        ],
        "decision": (
            "REJECT RUBRIC/MODEL: it retains every V14 candidate visual "
            "false positive and cannot enter a prospective rule."
        ),
        "artifact_sha256": {
            "gemma_output": sha256(gemma_path),
            "frozen_manual_evaluation": sha256(manual_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.gemma, args.manual)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
