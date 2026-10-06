#!/usr/bin/env python3
"""Evaluate contact-sheet VLM predictions against frozen manual judgments."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def score(label: str, confidence: float) -> float:
    confidence = min(1.0, max(0.0, float(confidence)))
    if label == "yes":
        return confidence
    if label == "no":
        return 1.0 - confidence
    return 0.5


def metrics(rows: list[dict], gold_key: str, result_key: str, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted = score(row["result"][result_key], row["result"]["confidence"]) >= threshold
        gold = bool(row[gold_key])
        if predicted and gold:
            tp += 1
        elif predicted:
            fp += 1
        elif gold:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "n": len(rows),
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rubric-version", default="contact_sheet_v3")
    parser.add_argument("--model")
    args = parser.parse_args()

    gold_by_item = {
        row["item_id"]: row for row in load_jsonl(args.manifest)
    }
    selected = [
        row
        for row in load_jsonl(args.predictions)
        if row.get("rubric_version") == args.rubric_version
        and (args.model is None or row.get("model") == args.model)
    ]
    missing_gold = sorted({row["item_id"] for row in selected} - gold_by_item.keys())
    if missing_gold:
        raise SystemExit(f"predictions absent from manifest: {missing_gold[:5]}")
    all_rows = [
        {
            **row,
            "gold_visual_content_present": gold_by_item[row["item_id"]][
                "gold_visual_content_present"
            ],
            "gold_social_behavior_scene": gold_by_item[row["item_id"]][
                "gold_social_behavior_scene"
            ],
            "gold_target_source_pass": gold_by_item[row["item_id"]][
                "gold_target_source_pass"
            ],
        }
        for row in selected
    ]
    rows = [row for row in all_rows if row.get("error") is None and row.get("result")]
    failures = [row for row in all_rows if row.get("error") is not None]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        groups["overall"].append(row)
        groups[f"pillar:{row['pillar']}"].append(row)
        groups[f"mode:{row['mode']}"].append(row)
        groups[f"mode:{row['mode']}:pillar:{row['pillar']}"].append(row)

    report = {
        "predictions": len(all_rows),
        "rubric_version": args.rubric_version,
        "model": args.model,
        "successful": len(rows),
        "failures": [
            {"item_id": row["item_id"], "mode": row["mode"], "error": row["error"]}
            for row in failures
        ],
        "tasks": {},
    }
    for task, gold_key, result_key, allowed_mode in (
        (
            "blind_social_behavior_scene",
            "gold_social_behavior_scene",
            "social_behavior_scene_visible",
            "blind",
        ),
        (
            "conditioned_target_source_pass",
            "gold_target_source_pass",
            "target_pillar_signal_visible",
            "conditioned",
        ),
    ):
        task_groups = {}
        for group, group_rows in sorted(groups.items()):
            eligible = [row for row in group_rows if row["mode"] == allowed_mode]
            if not eligible:
                continue
            task_groups[group] = [
                metrics(eligible, gold_key, result_key, threshold)
                for threshold in (0.5, 0.6, 0.7, 0.8, 0.9)
            ]
        report["tasks"][task] = task_groups
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"successful": len(rows), "failures": len(failures)}, sort_keys=True))


if __name__ == "__main__":
    main()
