#!/usr/bin/env python3
"""Evaluate label-blind episode extraction followed by symbolic label comparison."""

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
except ModuleNotFoundError:
    from evaluate_instructional_v22_intensive import metric  # type: ignore[no-redef]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_success(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        if row.get("error") in (None, "") and isinstance(row.get("result"), dict):
            output[str(row["item_id"])] = row
    return output


SOCIAL_BASES = {
    "interpersonal_treatment", "relationship_duty", "institutional_fairness",
    "public_shared_conduct", "care_or_welfare",
}


def episode_without_completeness(result: dict[str, Any]) -> bool:
    return (
        result["visual_context"] in {"connected_episode", "roleplay_episode"}
        and result["participant_grounding"] in {
            "affected_party_present", "shared_audience_present"
        }
        and (
            result["physical_social_action"] == "yes"
            or result["quote_function"] == "situated_to_present_party"
        )
    )


def label_with_relaxed_episode(
    episode: dict[str, Any], symbolic: dict[str, Any]
) -> bool:
    return (
        episode_without_completeness(episode)
        and symbolic["behavior_match"] == "yes"
        and symbolic["social_basis"] in SOCIAL_BASES
        and bool(symbolic["relation_matches_proposed_polarity"])
    )


def evaluate(
    semantic_rows: list[dict[str, Any]], ledger_rows: list[dict[str, str]],
    episode_outputs: dict[str, list[dict[str, Any]]],
    symbolic_outputs: dict[str, list[dict[str, Any]]], expected_count: int,
    semantic_ledger_rows: list[dict[str, str]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(episode_outputs) != set(symbolic_outputs) or not episode_outputs:
        raise ValueError("episode and symbolic model names must match")
    if len(semantic_rows) != expected_count or len(ledger_rows) != expected_count:
        raise ValueError("manual cohort coverage mismatch")
    semantics = {str(row["item_id"]): row for row in semantic_rows}
    semantics_by_index = {
        str(row["audit_index"]): row for row in semantic_rows
        if row.get("audit_index") is not None
    }
    if all(row.get("item_id") for row in ledger_rows):
        ledger = {str(row["item_id"]): row for row in ledger_rows}
    else:
        if len(semantics_by_index) != expected_count:
            raise ValueError("indexed manual ledger requires unique semantic audit_index")
        indexed_ledger = {str(row["audit_index"]): row for row in ledger_rows}
        if len(indexed_ledger) != expected_count:
            raise ValueError("duplicate or missing manual audit_index")
        ledger = {
            str(source["item_id"]): indexed_ledger[index]
            for index, source in semantics_by_index.items()
        }
    semantic_ledger: dict[str, dict[str, str]] | None = None
    if semantic_ledger_rows is not None:
        indexed_semantic_ledger = {
            str(row["audit_index"]): row for row in semantic_ledger_rows
        }
        if len(indexed_semantic_ledger) != len(semantic_ledger_rows):
            raise ValueError("duplicate semantic-ledger audit_index")
        semantic_ledger = {
            str(source["item_id"]): indexed_semantic_ledger[index]
            for index, source in semantics_by_index.items()
            if index in indexed_semantic_ledger
        }
    episodes = {name: latest_success(rows) for name, rows in episode_outputs.items()}
    symbols = {name: latest_success(rows) for name, rows in symbolic_outputs.items()}
    if len(semantics) != expected_count or len(ledger) != expected_count:
        raise ValueError("duplicate semantic or manual item_id")
    for name in episodes:
        if set(episodes[name]) != set(semantics) or set(symbols[name]) != set(semantics):
            raise ValueError(f"{name} model coverage mismatch")
    evaluated = []
    for item_id, source in semantics.items():
        manual = ledger[item_id]
        visual_demo = manual["visual_demo"] == "yes"
        if "usable_demo" in manual:
            usable_demo = manual["usable_demo"] == "yes"
            semantic_note = ""
        else:
            if semantic_ledger is None:
                raise ValueError(
                    "manual ledger lacks usable_demo; --semantic-ledger is required"
                )
            semantic_manual = semantic_ledger.get(item_id)
            if visual_demo and semantic_manual is None:
                raise ValueError("every visual positive requires a semantic audit row")
            usable_demo = bool(
                semantic_manual and semantic_manual["exact_usable"] == "yes"
            )
            semantic_note = semantic_manual.get("note", "") if semantic_manual else ""
        episode_results = {name: rows[item_id]["result"] for name, rows in episodes.items()}
        symbolic_results = {name: rows[item_id]["result"] for name, rows in symbols.items()}
        episode_decisions = {name: bool(value["episode_pass"]) for name, value in episode_results.items()}
        label_decisions = {name: bool(value["label_pass"]) for name, value in symbolic_results.items()}
        relaxed_episode_decisions = {
            name: episode_without_completeness(value)
            for name, value in episode_results.items()
        }
        relaxed_label_decisions = {
            name: label_with_relaxed_episode(episode_results[name], value)
            for name, value in symbolic_results.items()
        }
        evaluated.append({
            "item_id": item_id, "uid": source.get("uid"),
            "category": source.get("category") or "unknown",
            "polarity": source.get("polarity") or "unknown",
            "manual_visual_form": manual["visual_form"],
            "manual_visual_demo": visual_demo,
            "manual_usable_demo": usable_demo,
            "manual_note": manual.get("note", ""),
            "manual_semantic_note": semantic_note,
            "episode_results": episode_results, "symbolic_results": symbolic_results,
            "episode_decisions": episode_decisions, "label_decisions": label_decisions,
            "relaxed_episode_decisions": relaxed_episode_decisions,
            "relaxed_label_decisions": relaxed_label_decisions,
            "episode_consensus": all(episode_decisions.values()),
            "label_consensus": all(label_decisions.values()),
            "label_union": any(label_decisions.values()),
        })
    visual_gold = [row["manual_visual_demo"] for row in evaluated]
    usable_gold = [row["manual_usable_demo"] for row in evaluated]
    episode_rules = {
        name: metric(visual_gold, [row["episode_decisions"][name] for row in evaluated])
        for name in episodes
    }
    episode_rules["all_model_consensus"] = metric(visual_gold, [row["episode_consensus"] for row in evaluated])
    relaxed_episode_rules = {
        name: metric(
            visual_gold,
            [row["relaxed_episode_decisions"][name] for row in evaluated],
        ) for name in episodes
    }
    label_rules = {
        name: metric(usable_gold, [row["label_decisions"][name] for row in evaluated])
        for name in symbols
    }
    label_rules["all_model_consensus"] = metric(usable_gold, [row["label_consensus"] for row in evaluated])
    label_rules["any_model_union"] = metric(usable_gold, [row["label_union"] for row in evaluated])
    relaxed_label_rules = {
        name: metric(
            usable_gold,
            [row["relaxed_label_decisions"][name] for row in evaluated],
        ) for name in symbols
    }
    slices: dict[str, dict[str, Any]] = {}
    for field in ("polarity", "manual_visual_form", "category"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in evaluated:
            groups[str(row[field])].append(row)
        slices[field] = {
            value: metric(
                [row["manual_usable_demo"] for row in rows],
                [row["label_consensus"] for row in rows],
            ) for value, rows in sorted(groups.items())
        }
    return evaluated, {
        "kind": "instructional_blind_episode_symbolic_label_v1_development_evaluation",
        "policy": "post_hoc_development_only_no_keep_reject_or_corpus_mutation",
        "items": len(evaluated), "manual_coverage_complete": True,
        "manual_semantic_rows": len(semantic_ledger_rows or []),
        "manual_visual_demos": sum(visual_gold), "manual_usable_demos": sum(usable_gold),
        "episode_stage": episode_rules,
        "episode_without_completeness_development": relaxed_episode_rules,
        "label_stage": label_rules,
        "label_with_relaxed_episode_development": relaxed_label_rules,
        "slices": slices,
        "status": "development_cohort_not_eligible_for_promotion",
        "automatic_acceptance": False, "corpus_mutation_authorized": False,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--semantic-manifest", type=Path, required=True); parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--semantic-ledger", type=Path)
    parser.add_argument("--episode-output", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--symbolic-output", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--expected-count", type=int, required=True); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    episode_paths = {name: Path(path) for name, path in args.episode_output}; symbolic_paths = {name: Path(path) for name, path in args.symbolic_output}
    rows, summary = evaluate(
        read_jsonl(args.semantic_manifest), read_tsv(args.ledger),
        {name: read_jsonl(path) for name, path in episode_paths.items()},
        {name: read_jsonl(path) for name, path in symbolic_paths.items()}, args.expected_count,
        read_tsv(args.semantic_ledger) if args.semantic_ledger else None,
    )
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary["artifact_sha256"] = {
        "semantic_manifest": sha256(args.semantic_manifest), "ledger": sha256(args.ledger),
        "semantic_ledger": sha256(args.semantic_ledger) if args.semantic_ledger else None,
        "episode_outputs": {name: sha256(path) for name, path in episode_paths.items()},
        "symbolic_outputs": {name: sha256(path) for name, path in symbolic_paths.items()},
        "evaluated": sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
