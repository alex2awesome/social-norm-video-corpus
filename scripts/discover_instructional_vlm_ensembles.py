#!/usr/bin/env python3
"""Compare simple VLM ensembles on revealed instructional development labels."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def semantic_labels(path: Path) -> dict[int, bool]:
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return {int(row["audit_index"]): row["exact_original"] == "Y" for row in rows}


def latest(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in jsonl(path):
        if row.get("error") is None and isinstance(row.get("result"), dict):
            result[row["item_id"]] = row
    return result


def score(bits: list[tuple[bool, bool]]) -> dict[str, Any]:
    tp = sum(pred and truth for pred, truth in bits)
    fp = sum(pred and not truth for pred, truth in bits)
    tn = sum(not pred and not truth for pred, truth in bits)
    fn = sum(not pred and truth for pred, truth in bits)
    return {
        "selected": tp + fp,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def discover(
    selection_path: Path, semantic_path: Path, gemma_path: Path
) -> dict[str, Any]:
    selection = jsonl(selection_path)
    labels = semantic_labels(semantic_path)
    gemma = latest(gemma_path)
    predicates: dict[str, Callable[[bool, bool, bool], bool]] = {
        "qwen": lambda q, g, m: q,
        "glm": lambda q, g, m: g,
        "gemma": lambda q, g, m: m,
        "qwen_and_glm": lambda q, g, m: q and g,
        "qwen_and_gemma": lambda q, g, m: q and m,
        "glm_and_gemma": lambda q, g, m: g and m,
        "majority_2_of_3": lambda q, g, m: sum((q, g, m)) >= 2,
        "all_3": lambda q, g, m: q and g and m,
        "qwen_or_gemma": lambda q, g, m: q or m,
        "any_3": lambda q, g, m: q or g or m,
    }
    model_bits: list[tuple[int, bool, bool, bool, bool]] = []
    for row in selection:
        index = int(row["audit_index"])
        gemma_row = gemma.get(row["item_id"])
        if gemma_row is None:
            raise ValueError(f"missing Gemma record for {row['item_id']}")
        qwen = row["qwen_v11"]["result"]["exact_violation_demo"] == "yes"
        glm = row["glm_v11"]["result"]["exact_violation_demo"] == "yes"
        gemma_yes = gemma_row["result"]["exact_violation_demo"] == "yes"
        model_bits.append((index, qwen, glm, gemma_yes, labels[index]))
    rules: dict[str, Any] = {}
    for name, predicate in predicates.items():
        predicted = [
            (predicate(qwen, glm, gemma_yes), truth)
            for _, qwen, glm, gemma_yes, truth in model_bits
        ]
        summary = score(predicted)
        summary["false_positive_indices"] = [
            index
            for index, qwen, glm, gemma_yes, truth in model_bits
            if predicate(qwen, glm, gemma_yes) and not truth
        ]
        summary["false_negative_indices"] = [
            index
            for index, qwen, glm, gemma_yes, truth in model_bits
            if not predicate(qwen, glm, gemma_yes) and truth
        ]
        rules[name] = summary
    return {
        "kind": "instructional_vlm_ensemble_discovery",
        "development_items": len(model_bits),
        "rules": rules,
        "warning": "Revealed development comparison only; not a holdout result.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--semantic-ledger", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(
        json.dumps(
            discover(args.selection, args.semantic_ledger, args.gemma),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
