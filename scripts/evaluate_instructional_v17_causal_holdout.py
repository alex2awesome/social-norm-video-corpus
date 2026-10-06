#!/usr/bin/env python3
"""Evaluate the preregistered instructional V17 causal-transfer holdout."""

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


YES_NO = {"Y", "N"}
VISUAL_LABELS = {"Y", "N", "U"}
POLARITIES = {
    "violation",
    "correct",
    "neutral",
    "mixed",
    "none",
    "unclear",
    "described_only",
    "aftermath_only",
}
BANDS = {
    "candidate",
    "glm_v10a_reject",
    "qwen_v16_reject",
    "gemma_causal_reject",
}


def is_yes(value: str | None) -> bool:
    return str(value).strip().lower() in {"y", "yes", "true", "1"}


def normalize_yes_no(value: str | None) -> str | None:
    normalized = str(value).strip().lower()
    if normalized in {"y", "yes", "true", "1"}:
        return "Y"
    if normalized in {"n", "no", "false", "0"}:
        return "N"
    return None


def normalize_visual_label(value: str | None) -> str | None:
    normalized = normalize_yes_no(value)
    if normalized is not None:
        return normalized
    if str(value).strip().lower() in {"u", "uncertain", "unclear"}:
        return "U"
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
        visual_demo = normalize_visual_label(vrow.get("visual_demo"))
        conclusive = normalize_yes_no(vrow.get("visual_conclusive"))
        exact = normalize_yes_no(srow.get("exact_original"))
        relabel = normalize_yes_no(srow.get("usable_after_relabel"))
        polarity = srow.get("visible_polarity")
        failure = srow.get("failure_mechanism", "")
        if band not in BANDS:
            raise ValueError(f"audit_index {index}: invalid band {band!r}")
        if visual_demo not in VISUAL_LABELS or conclusive not in YES_NO:
            raise ValueError(f"audit_index {index}: invalid visual judgment")
        if exact not in YES_NO or relabel not in YES_NO:
            raise ValueError(f"audit_index {index}: invalid semantic Y/N")
        if polarity not in POLARITIES:
            raise ValueError(f"audit_index {index}: invalid visible_polarity")
        if exact == "Y" and (
            visual_demo != "Y" or relabel != "Y" or failure != "none"
        ):
            raise ValueError(
                f"audit_index {index}: exact item must be a visible, "
                "relabel-usable demo with failure=none"
            )
        if exact == "N" and failure in {"", "none"}:
            raise ValueError(
                f"audit_index {index}: inexact item needs a failure mechanism"
            )
        if visual_demo != "Y" and relabel != "N":
            raise ValueError(
                f"audit_index {index}: non-positive visual item cannot be "
                "label-usable"
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
                "manual_visual_demo_label": visual_demo,
                "manual_visual_uncertain": visual_demo == "U",
                "manual_visual_conclusive": conclusive == "Y",
                "manual_exact_original": exact == "Y",
                "manual_usable_after_relabel": relabel == "Y",
                "manual_visible_polarity": polarity,
                "manual_failure_mechanism": failure,
                "manual_corrected_event": srow.get("corrected_event"),
                "manual_notes": srow.get("manual_notes"),
                "manual_visual_form": vrow.get("visual_form"),
                "manual_visual_note": (
                    vrow.get("blind_visual_note") or vrow.get("visual_note")
                ),
                "protocol_exception": is_yes(vrow.get("protocol_exception")),
                "is_v17_candidate": band == "candidate",
            }
        )

    candidates = [row for row in joined if row["is_v17_candidate"]]
    controls = [row for row in joined if not row["is_v17_candidate"]]
    controls_by_band: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in controls:
        controls_by_band[row["band"]].append(row)

    visual_metric = metric(candidates, lambda row: row["manual_visual_demo"])
    exact_metric = metric(candidates, lambda row: row["manual_exact_original"])
    relabel_metric = metric(
        candidates, lambda row: row["manual_usable_after_relabel"]
    )
    distinct_uids = len({row["uid"] for row in candidates})
    visual_failures = Counter(
        row["manual_failure_mechanism"]
        for row in candidates
        if not row["manual_visual_demo"]
    )
    exact_failures = Counter(
        row["manual_failure_mechanism"]
        for row in candidates
        if not row["manual_exact_original"]
    )
    max_visual_failure = max(visual_failures.values(), default=0)
    checks = {
        "at_least_30_candidate_clips": len(candidates) >= 30,
        "at_least_20_distinct_candidate_uids": distinct_uids >= 20,
        "all_candidates_and_controls_conclusive": all(
            row["manual_visual_conclusive"] for row in joined
        ),
        "candidate_visual_demo_at_least_95_percent": (
            visual_metric["rate"] is not None
            and visual_metric["rate"] >= 0.95
        ),
        "candidate_exact_original_at_least_90_percent": (
            exact_metric["rate"] is not None
            and exact_metric["rate"] >= 0.90
        ),
        "candidate_usable_after_relabel_at_least_95_percent": (
            relabel_metric["rate"] is not None
            and relabel_metric["rate"] >= 0.95
        ),
        "no_repeated_visual_false_positive_mechanism_over_5_percent": (
            not candidates
            or max_visual_failure / len(candidates) <= 0.05
        ),
    }
    passed = all(checks.values())
    control_metrics = {
        band: {
            "visual_demo": metric(
                rows, lambda row: row["manual_visual_demo"]
            ),
            "exact_original": metric(
                rows, lambda row: row["manual_exact_original"]
            ),
            "usable_after_relabel": metric(
                rows, lambda row: row["manual_usable_after_relabel"]
            ),
            "false_reject_visual_count": sum(
                row["manual_visual_demo"] for row in rows
            ),
            "false_reject_usable_count": sum(
                row["manual_usable_after_relabel"] for row in rows
            ),
        }
        for band, rows in sorted(controls_by_band.items())
    }
    summary = {
        "kind": "instructional_v17_causal_transfer_dailymotion_validation",
        "coverage": {
            "audited_total": len(joined),
            "candidate_clips": len(candidates),
            "candidate_distinct_uids": distinct_uids,
            "control_clips": len(controls),
            "control_band_counts": dict(
                sorted(Counter(row["band"] for row in controls).items())
            ),
            "protocol_exceptions": sum(
                row["protocol_exception"] for row in joined
            ),
        },
        "candidate_metrics": {
            "visual_demo": visual_metric,
            "exact_original": exact_metric,
            "usable_after_relabel": relabel_metric,
        },
        "candidate_visual_false_positive_mechanisms": dict(
            sorted(visual_failures.items())
        ),
        "candidate_exact_failure_mechanisms": dict(
            sorted(exact_failures.items())
        ),
        "candidate_visual_false_positive_indices": [
            row["audit_index"]
            for row in candidates
            if not row["manual_visual_demo"]
        ],
        "candidate_inexact_indices": [
            row["audit_index"]
            for row in candidates
            if not row["manual_exact_original"]
        ],
        "control_metrics_by_rejection_band": control_metrics,
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
