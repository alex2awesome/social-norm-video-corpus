#!/usr/bin/env python3
"""Evaluate guarded OCR-line masks with abstentions in the cohort denominator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from scripts.evaluate_commentary_caption_crop_experiment import (
        evaluate,
        read_jsonl,
        read_tsv,
        sha256,
    )
else:
    from evaluate_commentary_caption_crop_experiment import (
        evaluate,
        read_jsonl,
        read_tsv,
        sha256,
    )


def evaluate_with_abstentions(manifest, blind, plan_summary, threshold=0.8):
    report, accepted = evaluate(manifest, blind)
    cohort = int(plan_summary["cohort_items"])
    renderable = int(plan_summary["renderable_items"])
    abstentions = list(plan_summary["abstentions"])
    if renderable != len(manifest):
        raise ValueError("plan renderable count does not match manifest")
    if cohort != renderable + len(abstentions):
        raise ValueError("cohort does not equal renderable items plus abstentions")
    usable = len(accepted)
    report.update({
        "kind": "commentary_ocr_line_mask_manual_evaluation_v2",
        "cohort_items": cohort,
        "renderable_items": renderable,
        "abstention_count": len(abstentions),
        "abstentions": abstentions,
        "cohort_usable_outputs": usable,
        "cohort_usable_rate": usable / cohort,
        "promotion_threshold": threshold,
        "shadow_transform_candidate_promoted": usable / cohort >= threshold,
        "automatic_keep_rule_promoted": False,
        "corpus_mutated": False,
    })
    return report, accepted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--plan-summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.accepted.exists():
        raise SystemExit("refusing to overwrite evaluation artifacts")
    report, accepted = evaluate_with_abstentions(
        read_jsonl(args.manifest), read_tsv(args.blind),
        json.loads(args.plan_summary.read_text()),
    )
    args.accepted.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in accepted))
    report.setdefault("artifact_sha256", {}).update({
        "manifest": sha256(args.manifest),
        "blind": sha256(args.blind),
        "plan_summary": sha256(args.plan_summary),
        "accepted": sha256(args.accepted),
    })
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
