#!/usr/bin/env python3
"""Evaluate scene-oriented instructional title cues on frozen manual cohorts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from math import sqrt
from pathlib import Path
from typing import Any


POSITIVE_CUES = {
    "dramatized_story": re.compile(
        r"dramatize|dramatized|dhar mann|short film|life lesson", re.I
    ),
    "social_experiment": re.compile(r"social experiment|what would you do", re.I),
    "animation_story": re.compile(
        r"animat|cartoon|toy|doll|story|friendship is magic", re.I
    ),
    "roleplay_scenario": re.compile(r"role.?play|scenario|skit|reenact", re.I),
    "psa": re.compile(r"\bpsa\b|public service", re.I),
}
NEGATIVE_CUE = re.compile(
    r"interview|podcast|webinar|lecture|tedx|tips|how to|explained|"
    r"discussion|panel|tutorial|news",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"yes", "y", "true", "1"}:
        return True
    if normalized in {"no", "n", "false", "0", ""}:
        return False
    raise ValueError(f"expected boolean-like value, got {value!r}")


def title_cues(title: str) -> dict[str, Any]:
    positive = [name for name, pattern in POSITIVE_CUES.items() if pattern.search(title)]
    negative = bool(NEGATIVE_CUE.search(title))
    return {
        "positive_cues": positive,
        "negative_presenter_cue": negative,
        "scene_title_candidate": bool(positive) and not negative,
    }


def load_cohort(spec: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    labels_path = root / spec["labels"]
    semantic_path = root / spec["semantic"]
    if labels_path.suffix == ".tsv":
        with labels_path.open() as handle:
            labels = list(csv.DictReader(handle, delimiter="\t"))
    else:
        labels = read_jsonl(labels_path)
    semantics = {str(row["item_id"]): row for row in read_jsonl(semantic_path)}
    if len(semantics) == 0:
        raise ValueError(f"empty semantic cohort: {spec['name']}")
    records = []
    for label in labels:
        item_id = str(label["item_id"])
        if item_id not in semantics:
            raise ValueError(f"missing semantic row for {item_id}")
        semantic = semantics[item_id]
        title = str(semantic.get("title") or "")
        records.append(
            {
                "cohort": str(spec["name"]),
                "item_id": item_id,
                "uid": str(label.get("uid") or semantic.get("uid") or ""),
                "title": title,
                "manual_visual_demo": as_bool(label[spec["visual_field"]]),
                "manual_exact_demo": as_bool(label[spec["exact_field"]]),
                **title_cues(title),
            }
        )
    return records


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def metric(records: list[dict[str, Any]], gold_field: str) -> dict[str, Any]:
    selected = [row for row in records if row["scene_title_candidate"]]
    positives = [row for row in records if row[gold_field]]
    tp = sum(row[gold_field] for row in selected)
    fp = len(selected) - tp
    fn = len(positives) - tp
    return {
        "n": len(records),
        "manual_positives": len(positives),
        "selected": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / len(selected) if selected else None,
        "precision_wilson_95": wilson(tp, len(selected)),
        "recall": tp / len(positives) if positives else None,
        "recall_wilson_95": wilson(tp, len(positives)),
    }


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_cohort: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_cohort.setdefault(row["cohort"], []).append(row)
    return {
        "kind": "instructional_scene_title_cue_manual_evaluation",
        "policy": "retrieval_and_manual_review_priority_only_never_a_label",
        "all_existing_media_preserved": True,
        "items": len(records),
        "unique_uids": len({row["uid"] for row in records}),
        "aggregate": {
            "visual_demo": metric(records, "manual_visual_demo"),
            "exact_demo": metric(records, "manual_exact_demo"),
        },
        "cohorts": {
            name: {
                "visual_demo": metric(rows, "manual_visual_demo"),
                "exact_demo": metric(rows, "manual_exact_demo"),
            }
            for name, rows in sorted(by_cohort.items())
        },
        "cue_counts": {
            name: sum(name in row["positive_cues"] for row in records)
            for name in POSITIVE_CUES
        },
        "negative_cue_count": sum(row["negative_presenter_cue"] for row in records),
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    root = args.config.resolve().parents[int(config.get("root_parents", 1))]
    records = []
    for spec in config["cohorts"]:
        records.extend(load_cohort(spec, root))
    item_ids = [row["item_id"] for row in records]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("cohorts are not item-disjoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    summary = evaluate(records)
    summary["artifact_sha256"] = {
        "config": sha256(args.config),
        "normalized_records": sha256(args.output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
