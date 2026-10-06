#!/usr/bin/env python3
"""Evaluate only the preregistered V16/V21 VLM consensus rules."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable


QUALIFYING_SCOPES = {
    "tacit_interpersonal",
    "shared_public",
    "explicit_social_etiquette",
}
QUALIFYING_ROLES = {
    "situated_scene",
    "roleplay_demo",
    "animation_or_story_demo",
    "text_dialogue_demo",
}


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


def strict(record: dict[str, Any]) -> bool:
    return record["result"].get("demo_usable") == "yes"


def transition_core(record: dict[str, Any]) -> bool:
    result = record["result"]
    atomic = (
        "performed_social_behavior",
        "actor_action_target_same_event",
        "affected_party_or_shared_setting_present",
        "socially_evaluable_without_metadata",
    )
    return (
        all(result.get(key) == "yes" for key in atomic)
        and result.get("social_scope") in QUALIFYING_SCOPES
        and result.get("scene_role") in QUALIFYING_ROLES
    )


Rule = Callable[[dict[str, dict[str, Any]]], bool]


RULES: dict[str, Rule] = {
    "qwen_v16_strict": lambda records: strict(records["qwen_v16"]),
    "glm_v16_strict": lambda records: strict(records["glm_v16"]),
    "v16_strict_consensus": lambda records: (
        strict(records["qwen_v16"]) and strict(records["glm_v16"])
    ),
    "v16_transition_core_consensus": lambda records: (
        transition_core(records["qwen_v16"])
        and transition_core(records["glm_v16"])
    ),
    "cross_rubric_consensus": lambda records: (
        transition_core(records["qwen_v16"])
        and transition_core(records["glm_v16"])
        and (
            transition_core(records["qwen_v21"])
            or transition_core(records["glm_v21"])
        )
    ),
    "v21_transition_core_consensus": lambda records: (
        transition_core(records["qwen_v21"])
        and transition_core(records["glm_v21"])
    ),
}


def target_metric(selected: list[dict[str, Any]], target: str) -> dict[str, Any]:
    positives = sum(bool(row[target]) for row in selected)
    return {
        "accepted": len(selected),
        "positive": positives,
        "negative": len(selected) - positives,
        "precision": positives / len(selected) if selected else None,
    }


def evaluate(
    manual_rows: list[dict[str, Any]],
    model_rows: dict[str, list[dict[str, Any]]],
    *,
    minimum_precision: float = 0.9,
    minimum_support: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(manual_rows) != 60:
        raise ValueError("expected frozen 60-row manual audit")
    keyed = {name: latest_success(rows) for name, rows in model_rows.items()}
    required = {"qwen_v16", "glm_v16", "qwen_v21", "glm_v21"}
    if set(keyed) != required:
        raise ValueError(f"model inputs must be exactly {sorted(required)}")

    evaluated: list[dict[str, Any]] = []
    for manual in manual_rows:
        item_id = str(manual["item_id"])
        records: dict[str, dict[str, Any]] = {}
        for name in sorted(required):
            if item_id not in keyed[name]:
                raise ValueError(f"missing {name} output for {item_id}")
            records[name] = keyed[name][item_id]
        decisions = {name: rule(records) for name, rule in RULES.items()}
        evaluated.append(
            {
                **manual,
                "consensus_rule_decisions": decisions,
                "model_outputs": {
                    name: {
                        "demo_usable": record["result"].get("demo_usable"),
                        "demo_usable_raw": record["result"].get(
                            "demo_usable_raw"
                        ),
                        "transition_core": transition_core(record),
                        "scene_role": record["result"].get("scene_role"),
                        "social_scope": record["result"].get("social_scope"),
                        "evidence": record["result"].get("evidence"),
                    }
                    for name, record in records.items()
                },
            }
        )

    results: dict[str, Any] = {}
    any_passed = False
    for rule_name in RULES:
        selected = [
            row
            for row in evaluated
            if row["consensus_rule_decisions"][rule_name]
        ]
        visual = target_metric(selected, "manual_visual_demo")
        usable = target_metric(selected, "manual_relabel_usable")
        exact = target_metric(selected, "manual_exact_original")
        passed = (
            len(selected) >= minimum_support
            and visual["precision"] is not None
            and visual["precision"] >= minimum_precision
            and usable["precision"] is not None
            and usable["precision"] >= minimum_precision
        )
        any_passed = any_passed or passed
        results[rule_name] = {
            "passed": passed,
            "visual": visual,
            "relabel_usable": usable,
            "exact_original": exact,
            "selected_indices": [row["audit_index"] for row in selected],
            "selected_manual_audit": [
                {
                    "audit_index": row["audit_index"],
                    "candidate_id": row["candidate_id"],
                    "manual_visual_demo": row["manual_visual_demo"],
                    "manual_relabel_usable": row["manual_relabel_usable"],
                    "manual_exact_original": row["manual_exact_original"],
                    "visual_form": row["visual_form"],
                    "manual_note": row["blind_visual_note"],
                    "model_evidence": {
                        name: output["evidence"]
                        for name, output in row["model_outputs"].items()
                    },
                }
                for row in selected
            ],
        }

    summary = {
        "kind": "instructional_v16_v21_preregistered_consensus_audit",
        "status": (
            "one_or_more_shadow_rules_passed"
            if any_passed
            else "all_rules_failed_no_promotion"
        ),
        "policy": "read_only_no_keep_reject_or_corpus_mutation",
        "coverage": {
            "manual_rows": len(manual_rows),
            **{
                f"{name}_successful": len(records)
                for name, records in keyed.items()
            },
        },
        "gate": {
            "minimum_support": minimum_support,
            "minimum_visual_precision": minimum_precision,
            "minimum_relabel_usable_precision": minimum_precision,
        },
        "rules": results,
    }
    return evaluated, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--qwen-v16", type=Path, required=True)
    parser.add_argument("--glm-v16", type=Path, required=True)
    parser.add_argument("--qwen-v21", type=Path, required=True)
    parser.add_argument("--glm-v21", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--minimum-precision", type=float, default=0.9)
    parser.add_argument("--minimum-support", type=int, default=5)
    args = parser.parse_args()
    sources = {
        "qwen_v16": args.qwen_v16,
        "glm_v16": args.glm_v16,
        "qwen_v21": args.qwen_v21,
        "glm_v21": args.glm_v21,
    }
    rows, summary = evaluate(
        read_jsonl(args.manual),
        {name: read_jsonl(path) for name, path in sources.items()},
        minimum_precision=args.minimum_precision,
        minimum_support=args.minimum_support,
    )
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary["artifact_sha256"] = {
        "manual": sha256(args.manual),
        "preregistration": sha256(args.preregistration),
        **{name: sha256(path) for name, path in sources.items()},
        "evaluated": sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
