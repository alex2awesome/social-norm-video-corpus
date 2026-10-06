#!/usr/bin/env python3
"""Evaluate decomposed instructional critics against a complete manual ledger."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_instructional_v22_intensive import metric
    from scripts.run_instructional_atomic_critic_v1 import (
        GROUNDED,
        SOCIAL_BASES,
        relation_matches_polarity,
    )
except ModuleNotFoundError:
    from evaluate_instructional_v22_intensive import metric  # type: ignore[no-redef]
    from run_instructional_atomic_critic_v1 import (  # type: ignore[no-redef]
        GROUNDED,
        SOCIAL_BASES,
        relation_matches_polarity,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def as_yes(value: str) -> bool:
    if value not in {"yes", "no"}:
        raise ValueError(f"expected yes/no, got {value!r}")
    return value == "yes"


def latest_success(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("error") in (None, "") and isinstance(row.get("result"), dict):
            output[str(row["item_id"])] = row
    return output


def ablation_pass(result: dict[str, Any], polarity: str, omit: str | None = None) -> bool:
    criteria = {
        "grounding": result["grounding"] in GROUNDED,
        "continuity": result["episode_continuity"] == "connected",
        "social_basis": result["social_basis"] in SOCIAL_BASES,
        "completeness": result["completeness"] == "complete",
        "relation": relation_matches_polarity(result["relation"], polarity),
    }
    return all(value for name, value in criteria.items() if name != omit)


def evaluate(
    semantic_rows: list[dict[str, Any]],
    ledger_rows: list[dict[str, str]],
    model_outputs: dict[str, list[dict[str, Any]]],
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(semantic_rows) != expected_count or len(ledger_rows) != expected_count:
        raise ValueError("semantic manifest and manual ledger must have expected coverage")
    semantics = {str(row["item_id"]): row for row in semantic_rows}
    ledger = {str(row["item_id"]): row for row in ledger_rows}
    if len(semantics) != expected_count or len(ledger) != expected_count:
        raise ValueError("duplicate semantic or ledger item_id")
    models = {name: latest_success(rows) for name, rows in model_outputs.items()}
    if not models:
        raise ValueError("at least one model output is required")
    for name, records in models.items():
        missing = sorted(set(semantics) - set(records))
        extra = sorted(set(records) - set(semantics))
        if missing or extra:
            raise ValueError(f"{name} coverage mismatch: missing={missing} extra={extra}")

    evaluated = []
    for item_id, source in semantics.items():
        if item_id not in ledger:
            raise ValueError(f"missing manual label: {item_id}")
        manual = ledger[item_id]
        polarity = str(source.get("polarity") or "").lower()
        results = {name: records[item_id]["result"] for name, records in models.items()}
        decisions = {name: bool(result["atomic_pass"]) for name, result in results.items()}
        evaluated.append(
            {
                "item_id": item_id,
                "uid": source.get("uid"),
                "category": source.get("category") or "unknown",
                "polarity": polarity,
                "manual_visual_form": manual["visual_form"],
                "manual_usable_demo": as_yes(manual["usable_demo"]),
                "manual_note": manual.get("note", ""),
                "model_results": results,
                "model_decisions": decisions,
                "all_model_consensus": all(decisions.values()),
                "any_model_union": any(decisions.values()),
                "ablations": {
                    name: {
                        omit: ablation_pass(result, polarity, omit)
                        for omit in (
                            "grounding", "continuity", "social_basis",
                            "completeness", "relation",
                        )
                    }
                    for name, result in results.items()
                },
            }
        )

    gold = [row["manual_usable_demo"] for row in evaluated]
    rules = {
        name: metric(gold, [row["model_decisions"][name] for row in evaluated])
        for name in models
    }
    rules["all_model_consensus"] = metric(
        gold, [row["all_model_consensus"] for row in evaluated]
    )
    rules["any_model_union"] = metric(gold, [row["any_model_union"] for row in evaluated])
    ablations = {
        name: {
            omit: metric(gold, [row["ablations"][name][omit] for row in evaluated])
            for omit in (
                "grounding", "continuity", "social_basis", "completeness", "relation"
            )
        }
        for name in models
    }
    fields = (
        "episode_medium", "grounding", "episode_continuity", "social_basis",
        "relation", "completeness",
    )
    agreement = {}
    if len(models) == 2:
        left, right = list(models)
        agreement = {
            field: sum(
                row["model_results"][left][field] == row["model_results"][right][field]
                for row in evaluated
            ) / len(evaluated)
            for field in fields
        }
    slices: dict[str, dict[str, Any]] = {}
    for field in ("polarity", "manual_visual_form", "category"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in evaluated:
            groups[str(row[field])].append(row)
        slices[field] = {
            value: metric(
                [row["manual_usable_demo"] for row in rows],
                [row["all_model_consensus"] for row in rows],
            )
            for value, rows in sorted(groups.items())
        }
    return evaluated, {
        "kind": "instructional_atomic_critic_v1_development_evaluation",
        "policy": "post_hoc_development_only_no_keep_reject_or_corpus_mutation",
        "items": len(evaluated),
        "manual_coverage_complete": True,
        "manual_usable_demos": sum(gold),
        "rules": rules,
        "single_field_ablations": ablations,
        "exact_field_agreement": agreement,
        "slices": slices,
        "status": "development_cohort_not_eligible_for_promotion",
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--model-output", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    output_paths = {name: Path(path) for name, path in args.model_output}
    evaluated, summary = evaluate(
        read_jsonl(args.semantic_manifest),
        read_tsv(args.ledger),
        {name: read_jsonl(path) for name, path in output_paths.items()},
        args.expected_count,
    )
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in evaluated))
    summary["artifact_sha256"] = {
        "semantic_manifest": sha256(args.semantic_manifest),
        "ledger": sha256(args.ledger),
        "model_outputs": {name: sha256(path) for name, path in output_paths.items()},
        "evaluated": sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
