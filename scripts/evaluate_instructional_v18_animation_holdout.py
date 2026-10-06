#!/usr/bin/env python3
"""Evaluate the preregistered instructional V18 animation-rule holdout."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_instructional_v11_holdout import (
        indexed,
        metric,
        read_jsonl,
        read_tsv,
        sha256,
    )
except ModuleNotFoundError:
    from evaluate_instructional_v11_holdout import (
        indexed,
        metric,
        read_jsonl,
        read_tsv,
        sha256,
    )


BANDS = {"candidate", "live_scene_control", "qwen_scene_reject_control"}
YES_NO = {"Y", "N"}


def normalize_yes_no(value: str | None) -> str | None:
    normalized = str(value).strip().lower()
    if normalized in {"y", "yes", "true", "1"}:
        return "Y"
    if normalized in {"n", "no", "false", "0"}:
        return "N"
    return None


def evaluate(
    selection_path: Path,
    visual_path: Path,
    semantic_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection_rows = read_jsonl(selection_path)
    expected = set(range(len(selection_rows)))
    selection = indexed(selection_rows, "selection", expected)
    visual = indexed(read_tsv(visual_path), "visual", expected)
    semantic = indexed(read_tsv(semantic_path), "semantic", expected)

    joined: list[dict[str, Any]] = []
    for index in range(len(selection_rows)):
        source = selection[index]
        vrow = visual[index]
        srow = semantic[index]
        band = str(source.get("band", ""))
        visual_demo = normalize_yes_no(vrow.get("visual_demo"))
        conclusive = normalize_yes_no(vrow.get("visual_conclusive"))
        exact = normalize_yes_no(srow.get("exact_social_norm"))
        relabel = normalize_yes_no(srow.get("relabel_usable"))
        failure = str(srow.get("failure_mode") or "")
        if band not in BANDS:
            raise ValueError(f"audit_index {index}: invalid band {band!r}")
        if (
            visual_demo not in YES_NO
            or conclusive not in YES_NO
            or exact not in YES_NO
            or relabel not in YES_NO
        ):
            raise ValueError(f"audit_index {index}: invalid or unresolved judgment")
        if exact == "Y" and (
            visual_demo != "Y" or relabel != "Y" or failure != "pass"
        ):
            raise ValueError(
                f"audit_index {index}: exact item must be a visible, "
                "relabel-usable demo with failure_mode=pass"
            )
        if exact == "N" and failure in {"", "pass"}:
            raise ValueError(
                f"audit_index {index}: inexact item needs a failure mode"
            )
        if visual_demo == "N" and relabel != "N":
            raise ValueError(
                f"audit_index {index}: visual non-demo cannot be relabel-usable"
            )
        joined.append(
            {
                "audit_index": index,
                "candidate_id": source["candidate_id"],
                "item_id": source["item_id"],
                "uid": source["uid"],
                "band": band,
                "category": source.get("category"),
                "norm": source.get("norm"),
                "polarity": source.get("polarity"),
                "manual_visual_demo": visual_demo == "Y",
                "manual_visual_conclusive": conclusive == "Y",
                "manual_exact_social_norm": exact == "Y",
                "manual_relabel_usable": relabel == "Y",
                "manual_failure_mode": failure,
                "manual_corrected_event": srow.get("corrected_event"),
                "manual_semantic_note": srow.get("semantic_note"),
                "manual_visual_form": vrow.get("visual_form"),
                "manual_visual_note": vrow.get("blind_visual_note"),
                "is_v18_candidate": band == "candidate",
            }
        )

    candidates = [row for row in joined if row["is_v18_candidate"]]
    controls_by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        if not row["is_v18_candidate"]:
            controls_by_band[row["band"]].append(row)

    candidate_metrics = {
        "visual_demo": metric(candidates, lambda row: row["manual_visual_demo"]),
        "exact_social_norm": metric(
            candidates, lambda row: row["manual_exact_social_norm"]
        ),
        "relabel_usable": metric(
            candidates, lambda row: row["manual_relabel_usable"]
        ),
    }
    visual_failures = Counter(
        row["manual_failure_mode"]
        for row in candidates
        if not row["manual_visual_demo"]
    )
    exact_failures = Counter(
        row["manual_failure_mode"]
        for row in candidates
        if not row["manual_exact_social_norm"]
    )
    max_visual_failure = max(visual_failures.values(), default=0)
    checks = {
        "exactly_30_candidate_clips": len(candidates) == 30,
        "all_60_source_uids_distinct": len({row["uid"] for row in joined}) == 60,
        "all_candidates_and_controls_conclusive": all(
            row["manual_visual_conclusive"] for row in joined
        ),
        "candidate_visual_demo_at_least_95_percent": (
            candidate_metrics["visual_demo"]["rate"] is not None
            and candidate_metrics["visual_demo"]["rate"] >= 0.95
        ),
        "candidate_exact_social_norm_at_least_90_percent": (
            candidate_metrics["exact_social_norm"]["rate"] is not None
            and candidate_metrics["exact_social_norm"]["rate"] >= 0.90
        ),
        "candidate_relabel_usable_at_least_95_percent": (
            candidate_metrics["relabel_usable"]["rate"] is not None
            and candidate_metrics["relabel_usable"]["rate"] >= 0.95
        ),
        "no_repeated_visual_false_positive_mechanism_over_5_percent": (
            not candidates
            or max_visual_failure / len(candidates) <= 0.05
        ),
    }
    control_metrics = {
        band: {
            "visual_demo": metric(rows, lambda row: row["manual_visual_demo"]),
            "exact_social_norm": metric(
                rows, lambda row: row["manual_exact_social_norm"]
            ),
            "relabel_usable": metric(
                rows, lambda row: row["manual_relabel_usable"]
            ),
        }
        for band, rows in sorted(controls_by_band.items())
    }
    passed = all(checks.values())
    summary = {
        "kind": "instructional_v18_animation_holdout_evaluation",
        "coverage": {
            "audited_total": len(joined),
            "candidate_clips": len(candidates),
            "control_band_counts": dict(
                sorted(Counter(row["band"] for row in joined if not row["is_v18_candidate"]).items())
            ),
            "distinct_uids": len({row["uid"] for row in joined}),
        },
        "candidate_metrics": candidate_metrics,
        "control_metrics_by_band": control_metrics,
        "candidate_visual_false_positive_mechanisms": dict(
            sorted(visual_failures.items())
        ),
        "candidate_exact_failure_mechanisms": dict(
            sorted(exact_failures.items())
        ),
        "candidate_visual_false_positive_indices": [
            row["audit_index"] for row in candidates if not row["manual_visual_demo"]
        ],
        "candidate_inexact_indices": [
            row["audit_index"]
            for row in candidates
            if not row["manual_exact_social_norm"]
        ],
        "preregistered_checks": checks,
        "preregistered_pass": passed,
        "promotion_eligible": passed,
        "platform_scope": "dailymotion_shadow_ranking_only",
        "decision": (
            "PASS AS NON-DESTRUCTIVE DAILYMOTION SHADOW RANKING ONLY."
            if passed
            else "FAIL: do not promote or mutate corpus state."
        ),
        "artifact_sha256": {
            "selection": sha256(selection_path),
            "manual_visual_ledger": sha256(visual_path),
            "manual_semantic_ledger": sha256(semantic_path),
        },
    }
    return joined, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--visual-ledger", type=Path, required=True)
    parser.add_argument("--semantic-ledger", type=Path, required=True)
    parser.add_argument("--evaluated-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()
    joined, summary = evaluate(
        args.selection, args.visual_ledger, args.semantic_ledger
    )
    args.evaluated_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in joined)
    )
    args.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
