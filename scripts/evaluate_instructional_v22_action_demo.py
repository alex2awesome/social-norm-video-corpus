#!/usr/bin/env python3
"""Evaluate V22 action-demo outputs against a frozen blind manual ledger."""

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
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_success(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {}
    for row in rows:
        if row.get("error") in (None, "") and isinstance(row.get("result"), dict):
            output[str(row["item_id"])] = row
    return output


def metric(labels: list[bool], predictions: list[bool]) -> dict[str, Any]:
    tp = sum(y and p for y, p in zip(labels, predictions))
    fp = sum(not y and p for y, p in zip(labels, predictions))
    fn = sum(y and not p for y, p in zip(labels, predictions))
    tn = sum(not y and not p for y, p in zip(labels, predictions))
    return {
        "n": len(labels),
        "selected": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def evaluate(
    manifest: list[dict[str, Any]],
    ledger: list[dict[str, str]],
    model_rows: dict[str, list[dict[str, Any]]],
    expected_count: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(manifest) != expected_count or len(ledger) != expected_count:
        raise ValueError(f"expected {expected_count} manifest and ledger rows")
    by_index = {int(row["audit_index"]): row for row in ledger}
    if len(by_index) != expected_count:
        raise ValueError("duplicate audit index")
    models = {name: latest_success(rows) for name, rows in model_rows.items()}
    if not models:
        raise ValueError("at least one model output is required")
    evaluated = []
    for source in manifest:
        index = int(source["audit_index"])
        manual = by_index[index]
        if manual["candidate_id"] != source["candidate_id"]:
            raise ValueError(f"candidate mismatch at {index}")
        decisions = {}
        outputs = {}
        for name, records in models.items():
            item_id = str(source["item_id"])
            if item_id not in records:
                raise ValueError(f"missing {name} output for {item_id}")
            result = records[item_id]["result"]
            decisions[name] = result.get("demo_pass") == "yes"
            outputs[name] = result
        values = list(decisions.values())
        evaluated.append(
            {
                **source,
                "manual_visual_demo": manual["visual_demo"].strip().lower() in {"y", "yes"},
                "manual_visual_form": manual["visual_form"],
                "manual_note": manual["blind_visual_note"],
                "model_decisions": decisions,
                "all_model_consensus": all(values),
                "any_model_positive": any(values),
                "model_outputs": outputs,
            }
        )
    labels = [row["manual_visual_demo"] for row in evaluated]
    rules = {
        **{
            name: metric(labels, [row["model_decisions"][name] for row in evaluated])
            for name in models
        },
        "all_model_consensus": metric(
            labels, [row["all_model_consensus"] for row in evaluated]
        ),
        "any_model_positive": metric(
            labels, [row["any_model_positive"] for row in evaluated]
        ),
    }
    return evaluated, {
        "kind": "instructional_v22_action_demo_development_audit",
        "status": "development_only_no_promotion",
        "policy": "read_only_shadow_no_keep_reject_or_corpus_mutation",
        "items": len(evaluated),
        "manual_visual_demos": sum(labels),
        "rules": rules,
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--model-output", action="append", nargs=2, metavar=("NAME", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=30)
    args = parser.parse_args()
    sources = {name: Path(path) for name, path in args.model_output}
    rows, summary = evaluate(
        read_jsonl(args.manifest),
        read_tsv(args.ledger),
        {name: read_jsonl(path) for name, path in sources.items()},
        args.expected_count,
    )
    args.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary["artifact_sha256"] = {
        "manifest": sha256(args.manifest),
        "ledger": sha256(args.ledger),
        "model_outputs": {name: sha256(path) for name, path in sources.items()},
        "evaluated": sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
