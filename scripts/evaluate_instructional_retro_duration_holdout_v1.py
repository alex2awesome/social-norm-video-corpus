#!/usr/bin/env python3
"""Evaluate every item in the independent retro-duration holdout."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import sqrt
from pathlib import Path
from typing import Any


TRINARY = {"yes", "no", "uncertain"}
VISUAL_FORMS = {
    "live_scene", "roleplay", "film_tv", "animation", "puppets", "gameplay",
    "demonstration", "talking_head", "lecture", "interview", "panel",
    "broll", "graphic", "text", "technical_demo", "mixed", "other", "uncertain",
}
GROUNDING = {"actor_and_recipient", "multiple_participants", "actor_only", "none", "uncertain"}
POLARITY = {"violation", "correct", "contrast", "mixed", "unclear", "not_applicable"}
FAILURE_MODES = {
    "none", "no_visual_demo", "reported_not_enacted", "disconnected_broll",
    "label_mismatch", "label_too_abstract", "off_topic", "technical_only",
    "incomplete_demo", "uncertain",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def keyed(rows: list[dict[str, Any]], name: str) -> dict[str, dict[str, Any]]:
    output = {str(row.get("candidate_id") or ""): row for row in rows}
    if "" in output or len(output) != len(rows):
        raise ValueError(f"{name} has empty or duplicate candidate_id")
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if not n:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def validate(
    sealed: list[dict[str, Any]], blind: list[dict[str, str]], post: list[dict[str, str]],
    prereg: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    media, visuals, semantics = keyed(sealed, "sealed manifest"), keyed(blind, "blind ledger"), keyed(post, "post-reveal ledger")
    if not (set(media) == set(visuals) == set(semantics)):
        raise ValueError("manual ledgers do not exactly cover the sealed holdout")
    if len(media) != int(prereg["items"]):
        raise ValueError("manual cohort size does not match preregistration")
    if len({str(row["uid"]) for row in sealed}) != len(sealed):
        raise ValueError("holdout is not source-disjoint")
    for candidate_id in media:
        visual, semantic = visuals[candidate_id], semantics[candidate_id]
        for field in ("visual_demo", "complete_demo"):
            if visual.get(field) not in TRINARY:
                raise ValueError(f"{candidate_id}: invalid or missing {field}")
        if visual.get("visual_form") not in VISUAL_FORMS:
            raise ValueError(f"{candidate_id}: invalid visual_form")
        if visual.get("participant_grounding") not in GROUNDING:
            raise ValueError(f"{candidate_id}: invalid participant_grounding")
        if not visual.get("blind_evidence", "").strip():
            raise ValueError(f"{candidate_id}: missing blind_evidence")
        for field in (
            "label_is_concrete_social_behavior", "visual_matches_original_label",
            "usable_after_relabel",
        ):
            if semantic.get(field) not in TRINARY:
                raise ValueError(f"{candidate_id}: invalid or missing {field}")
        if semantic.get("visible_polarity") not in POLARITY:
            raise ValueError(f"{candidate_id}: invalid visible_polarity")
        if semantic.get("failure_mode") not in FAILURE_MODES:
            raise ValueError(f"{candidate_id}: invalid failure_mode")
        if not semantic.get("post_reveal_evidence", "").strip():
            raise ValueError(f"{candidate_id}: missing post_reveal_evidence")
        if visual["complete_demo"] == "yes" and visual["visual_demo"] != "yes":
            raise ValueError(f"{candidate_id}: non-demo cannot be a complete demo")
        if (
            semantic["visual_matches_original_label"] == "yes"
            and semantic["label_is_concrete_social_behavior"] != "yes"
        ):
            raise ValueError(f"{candidate_id}: abstract label cannot exactly match")
        if semantic["usable_after_relabel"] == "yes" and visual["visual_demo"] != "yes":
            raise ValueError(f"{candidate_id}: non-demo cannot be usable after relabel")
        if (
            semantic["usable_after_relabel"] == "yes"
            and semantic["visual_matches_original_label"] != "yes"
            and not semantic.get("corrected_behavior", "").strip()
        ):
            raise ValueError(f"{candidate_id}: relabel-usable item needs corrected_behavior")
    return media, visuals, semantics


def group_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    visual = sum(row["manual_visual_demo"] for row in rows)
    usable = sum(row["manual_usable_demo"] for row in rows)
    return {
        "items": len(rows),
        "visual_demos": visual,
        "visual_demo_precision": visual / len(rows) if rows else None,
        "visual_demo_wilson_95": wilson(visual, len(rows)),
        "usable_demos_after_relabel": usable,
        "usable_demo_rate": usable / len(rows) if rows else None,
    }


def evaluate(sealed, blind, post, prereg):
    media, visuals, semantics = validate(sealed, blind, post, prereg)
    rows = []
    for candidate_id, item in media.items():
        visual, semantic = visuals[candidate_id], semantics[candidate_id]
        rows.append({
            "candidate_id": candidate_id,
            "uid": item["uid"],
            "cohort": item["holdout_cohort"],
            "manual_visual_demo": visual["visual_demo"] == "yes",
            "manual_complete_demo": visual["complete_demo"] == "yes",
            "manual_exact_original": semantic["visual_matches_original_label"] == "yes",
            "manual_usable_demo": (
                visual["visual_demo"] == "yes" and semantic["usable_after_relabel"] == "yes"
            ),
            "visual_form": visual["visual_form"],
            "failure_mode": semantic["failure_mode"],
        })
    cohorts = {
        name: group_metrics([row for row in rows if row["cohort"] == name])
        for name in sorted(set(row["cohort"] for row in rows))
    }
    signal = cohorts["signal_positive"]
    gate = prereg["gate"]
    checks = {
        "manual_review_complete": len(rows) == int(prereg["items"]),
        "source_disjoint": len({row["uid"] for row in rows}) == len(rows),
        "minimum_signal_positive_items": signal["items"] >= int(gate["minimum_signal_positive_items"]),
        "minimum_visual_demo_precision": signal["visual_demo_precision"] >= float(gate["minimum_visual_demo_precision"]),
        "minimum_visual_demo_wilson_95_lower": signal["visual_demo_wilson_95"][0] >= float(gate["minimum_visual_demo_wilson_95_lower"]),
    }
    passed = all(checks.values())
    return {
        "kind": "instructional_retro_duration_independent_holdout_evaluation_v1",
        "items": len(rows),
        "manual_review_complete": True,
        "cohorts": cohorts,
        "gate_checks": checks,
        "review_priority_rule_promoted": passed,
        "allowed_use": "manual_review_priority_only" if passed else "none",
        "automatic_acceptance": False,
        "corpus_mutated": False,
    }, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post-reveal", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.records.exists():
        raise SystemExit("refusing to overwrite evaluation artifacts")
    report, rows = evaluate(
        read_jsonl(args.sealed), read_tsv(args.blind), read_tsv(args.post_reveal),
        json.loads(args.preregistration.read_text()),
    )
    args.records.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report["artifact_sha256"] = {
        "sealed": sha256(args.sealed), "blind": sha256(args.blind),
        "post_reveal": sha256(args.post_reveal),
        "preregistration": sha256(args.preregistration), "records": sha256(args.records),
    }
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
