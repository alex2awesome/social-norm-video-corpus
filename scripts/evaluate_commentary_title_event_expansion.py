#!/usr/bin/env python3
"""Evaluate title-event VLMs on the fresh commentary expansion audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_success(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("error") in (None, "") and isinstance(row.get("result"), dict):
            output[int(row["audit_index"])] = row
    return output


def metrics(gold: list[bool], predicted: list[bool]) -> dict[str, Any]:
    tp = sum(y and p for y, p in zip(gold, predicted, strict=True))
    fp = sum(not y and p for y, p in zip(gold, predicted, strict=True))
    fn = sum(y and not p for y, p in zip(gold, predicted, strict=True))
    tn = sum(not y and not p for y, p in zip(gold, predicted, strict=True))
    return {
        "n": len(gold),
        "selected": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def decisions(result: dict[str, Any]) -> dict[str, bool]:
    candidate = result.get("candidate_event")
    bounded = (
        isinstance(result.get("candidate_start_sec"), (int, float))
        and not isinstance(result.get("candidate_start_sec"), bool)
        and isinstance(result.get("candidate_end_sec"), (int, float))
        and not isinstance(result.get("candidate_end_sec"), bool)
        and result["candidate_end_sec"] > result["candidate_start_sec"]
    )
    return {
        "candidate_yes": candidate == "yes",
        "candidate_or_bounded_uncertain": candidate == "yes"
        or (candidate == "uncertain" and bounded),
        "strict_literal": (
            candidate == "yes"
            and result.get("title_action_visible") == "yes"
            and result.get("actor_action_target_same_event") == "yes"
        ),
    }


def evaluate(
    ledger: list[dict[str, str]],
    model_rows: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manual = {int(row["audit_index"]): row for row in ledger}
    if len(manual) != len(ledger):
        raise ValueError("duplicate manual audit index")
    evaluable = {
        index: row
        for index, row in manual.items()
        if row["render_status"] == "ok"
        and row["source_has_candidate_event_footage"] in {"yes", "no", "uncertain"}
        and row["title_action_visible"] in {"yes", "no", "uncertain"}
    }
    models = {name: latest_success(rows) for name, rows in model_rows.items()}
    if not models:
        raise ValueError("at least one model output is required")
    for name, rows in models.items():
        missing = sorted(set(evaluable) - set(rows))
        if missing:
            raise ValueError(f"missing {name} output for audit indices {missing}")

    evaluated = []
    for index, gold in sorted(evaluable.items()):
        per_model = {
            name: decisions(rows[index]["result"])
            for name, rows in models.items()
        }
        evaluated.append(
            {
                "audit_index": index,
                "candidate_id": gold["candidate_id"],
                "gold_event_footage": gold["source_has_candidate_event_footage"],
                "gold_title_action": gold["title_action_visible"],
                "model_decisions": per_model,
                "model_outputs": {
                    name: rows[index]["result"] for name, rows in models.items()
                },
            }
        )

    targets = {
        "event_footage_yes": [row["gold_event_footage"] == "yes" for row in evaluated],
        "title_action_yes": [row["gold_title_action"] == "yes" for row in evaluated],
    }
    rules: dict[str, dict[str, dict[str, Any]]] = {}
    for decision_name in (
        "candidate_yes",
        "candidate_or_bounded_uncertain",
        "strict_literal",
    ):
        selections: dict[str, list[bool]] = {
            name: [row["model_decisions"][name][decision_name] for row in evaluated]
            for name in models
        }
        matrix = list(selections.values())
        if len(matrix) >= 2:
            selections["all_model_consensus"] = [all(values) for values in zip(*matrix)]
            selections["any_model_positive"] = [any(values) for values in zip(*matrix)]
        rules[decision_name] = {
            selector: {
                target: metrics(gold, predictions)
                for target, gold in targets.items()
            }
            for selector, predictions in selections.items()
        }
    return evaluated, {
        "kind": "commentary_title_event_fresh_expansion_evaluation",
        "status": "development_only_no_promotion",
        "policy": "localization_review_only_no_keep_reject",
        "manual_rows": len(manual),
        "evaluable_rendered_rows": len(evaluated),
        "explicit_render_failures": len(manual) - len(evaluated),
        "manual_event_footage_yes": sum(targets["event_footage_yes"]),
        "manual_title_action_yes": sum(targets["title_action_yes"]),
        "rules": rules,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument(
        "--model-output",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    sources = {name: Path(path) for name, path in args.model_output}
    rows, summary = evaluate(
        read_tsv(args.ledger),
        {name: read_jsonl(path) for name, path in sources.items()},
    )
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary["artifact_sha256"] = {
        "ledger": sha256(args.ledger),
        "model_outputs": {name: sha256(path) for name, path in sources.items()},
        "evaluated": sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
