#!/usr/bin/env python3
"""Evaluate semantic text atoms and exploratory visual conjunctions.

The 60-item mechanism cohort and source-disjoint 40-item uniform cohort are
reported separately. The static visual classifier is fit only on mechanism
labels. Semantic conjunctions are explicitly exploratory because they were
formulated after inspecting the first uniform-holdout results; they require a
new confirmation sample before promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_full_corpus_audit_features import load_gold
    from scripts.evaluate_full_corpus_multimodal_features import (
        flatten,
        last_successful_rows,
    )
    from scripts.evaluate_frozen_signal_conjunctions import (
        fitted_probabilities,
        rule_metrics,
    )
else:
    from evaluate_full_corpus_audit_features import load_gold
    from evaluate_full_corpus_multimodal_features import (
        flatten,
        last_successful_rows,
    )
    from evaluate_frozen_signal_conjunctions import (
        fitted_probabilities,
        rule_metrics,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def binary_metrics(
    item_ids: list[str],
    truth: dict[str, int],
    selected: dict[str, bool],
) -> dict[str, Any]:
    return rule_metrics(item_ids, truth, selected)


def manual_rows(audit_root: Path, split: str) -> dict[str, dict[str, Any]]:
    path = (
        audit_root
        / split
        / "instructional"
        / "manual_post_reveal_review.jsonl"
    )
    return {row["item_id"]: row for row in read_jsonl(path)}


def social_truth(
    rows: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, int]]:
    eligible = sorted(
        item_id
        for item_id, row in rows.items()
        if row["assigned_norm_is_social_norm"] in {"yes", "no"}
    )
    return (
        eligible,
        {
            item_id: int(
                rows[item_id]["assigned_norm_is_social_norm"] == "yes"
            )
            for item_id in eligible
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path, action="append")
    parser.add_argument("--atoms", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    atom_rows = {
        row["item_id"]: row
        for row in read_jsonl(args.atoms)
        if row.get("error") is None
    }
    if len(atom_rows) != 100:
        raise ValueError(f"expected 100 successful atom rows, got {len(atom_rows)}")
    atom_candidate = {
        item_id: bool(row["derived_social_candidate"])
        for item_id, row in atom_rows.items()
    }

    mechanism_manual = manual_rows(args.audit_root, "mechanism")
    uniform_manual = manual_rows(args.audit_root, "uniform")
    social_results: dict[str, Any] = {}
    for split, rows in (
        ("mechanism", mechanism_manual),
        ("uniform", uniform_manual),
    ):
        ids, truth = social_truth(rows)
        social_results[split] = binary_metrics(ids, truth, atom_candidate)

    merged = last_successful_rows(args.features)
    flat = {item_id: flatten(row) for item_id, row in merged.items()}
    train = load_gold(args.audit_root, "mechanism", "instructional")
    uniform = load_gold(args.audit_root, "uniform", "instructional")
    train_ids = sorted(train)
    uniform_ids = sorted(uniform)
    static_probability, static_meta = fitted_probabilities(
        train_ids,
        uniform_ids,
        flat,
        {item_id: int(values["visual"]) for item_id, values in train.items()},
        ("low.", "pose.", "clip."),
    )
    static = {
        item_id: static_probability[item_id] >= 0.5
        for item_id in uniform_ids
    }
    conjunction = {
        item_id: static[item_id] and atom_candidate[item_id]
        for item_id in uniform_ids
    }
    target_truths = {
        "visual_demo": {
            item_id: int(uniform[item_id]["visual"])
            for item_id in uniform_ids
        },
        "visual_demo_and_social_norm": {
            item_id: int(
                uniform_manual[item_id]["visual_demo_present"] == "yes"
                and uniform_manual[item_id]["assigned_norm_is_social_norm"]
                == "yes"
            )
            for item_id in uniform_ids
        },
        "strict_instructional": {
            item_id: int(uniform[item_id]["strict"])
            for item_id in uniform_ids
        },
    }
    exploratory: dict[str, Any] = {}
    for target_name, truth in target_truths.items():
        exploratory[target_name] = {
            "static_visual_at_0_5": binary_metrics(
                uniform_ids, truth, static
            ),
            "static_visual_and_social_atoms": binary_metrics(
                uniform_ids, truth, conjunction
            ),
        }

    report = {
        "protocol": {
            "semantic_atoms": (
                "mechanism_60_and_source_disjoint_uniform_40_reported_separately"
            ),
            "visual_model": "fit_mechanism_60_test_uniform_40",
            "conjunction_status": (
                "post_hoc_exploratory_requires_new_confirmation_sample"
            ),
            "policy": "shadow_only_no_keep_or_reject_decision",
        },
        "atom_rubric_versions": sorted(
            {row["rubric_version"] for row in atom_rows.values()}
        ),
        "semantic_social_norm_target": social_results,
        "static_visual_model": static_meta,
        "exploratory_uniform_conjunctions": exploratory,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
