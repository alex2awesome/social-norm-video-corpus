#!/usr/bin/env python3
"""Evaluate the text-only social-norm routing prior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate_visual_scene_scores import metrics  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = {row["item_id"]: row for row in load_jsonl(args.manifest)}
    rows = [
        row
        for row in load_jsonl(args.scores)
        if row.get("result") and row["item_id"] in manifest
    ]
    labels = [manifest[row["item_id"]].get("is_social_norm") == "yes" for row in rows]
    predictions = [
        row["result"]["social_norm_candidate"] == "yes" for row in rows
    ]
    strict_predictions = [
        row["result"]["social_norm_candidate"] == "yes"
        and row["result"].get("concrete_behavior_named") == "yes"
        and row["result"].get("affected_other_or_shared_setting_named") == "yes"
        and row["result"].get("norm_type")
        in {"tacit_interpersonal", "shared_public"}
        for row in rows
    ]
    report = metrics(labels, predictions)
    report["items_valid"] = len(rows)
    report["strict_operational"] = metrics(labels, strict_predictions)
    report["errors_total"] = sum(
        bool(row.get("error")) for row in load_jsonl(args.scores)
    )
    report["by_pillar"] = {}
    for pillar in sorted({row["pillar"] for row in rows}):
        indexes = [i for i, row in enumerate(rows) if row["pillar"] == pillar]
        report["by_pillar"][pillar] = metrics(
            [labels[i] for i in indexes],
            [predictions[i] for i in indexes],
        )
    report["errors"] = [
        {
            "item_id": row["item_id"],
            "pillar": row["pillar"],
            "gold": labels[index],
            "predicted": predictions[index],
            "norm": manifest[row["item_id"]].get("norm")
            or manifest[row["item_id"]].get("normalized_behavior"),
            "norm_type": row["result"].get("norm_type"),
            "reason": row["result"].get("reason"),
            "manual_description": manifest[row["item_id"]].get("description"),
        }
        for index, row in enumerate(rows)
        if labels[index] != predictions[index]
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
