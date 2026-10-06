#!/usr/bin/env python3
"""Evaluate authority-reaction cue versions on frozen manual gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def metrics(gold: list[bool], predicted: list[bool]) -> dict[str, Any]:
    tp = sum(g and p for g, p in zip(gold, predicted))
    fp = sum(not g and p for g, p in zip(gold, predicted))
    fn = sum(g and not p for g, p in zip(gold, predicted))
    tn = sum(not g and not p for g, p in zip(gold, predicted))
    return {
        "n": len(gold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / len(gold) if gold else 0.0,
    }


def evaluate(
    sealed: list[dict[str, Any]],
    post: list[dict[str, Any]],
    blind: list[dict[str, Any]],
    score_sets: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    post_by_index = {int(row["audit_index"]): row for row in post}
    blind_by_index = {int(row["audit_index"]): row for row in blind}
    if len(sealed) != len(post_by_index) or len(sealed) != len(blind_by_index):
        raise ValueError("sealed, blind, and post-reveal counts differ")
    gold = [
        post_by_index[int(row["audit_index"])]["authority_or_host_reaction"]
        == "yes"
        for row in sealed
    ]
    output: dict[str, Any] = {
        "kind": "witnessed_authority_reaction_cue_manual_benchmark",
        "items": len(sealed),
        "gold_authority_or_host_reactions": sum(gold),
        "blind_visual_scene_counts": {
            label: sum(
                blind_by_index[int(row["audit_index"])]["visual_scene_candidate"]
                == label
                for row in sealed
            )
            for label in ("yes", "uncertain", "no")
        },
        "models": {},
        "decision": {
            "automatic_promotion": False,
            "reason": (
                "A single source-disjoint match/miss audit can reject a weak "
                "rule; promotion requires a fresh independent confirmation."
            ),
        },
    }
    for name, scores in score_sets.items():
        by_item = {row["item_id"]: row for row in scores}
        missing = [row["item_id"] for row in sealed if row["item_id"] not in by_item]
        if missing:
            raise ValueError(f"{name}: missing {len(missing)} selected items")
        predicted = [
            bool(by_item[row["item_id"]]["authority_reaction_cue"])
            for row in sealed
        ]
        model_metrics = metrics(gold, predicted)
        model_metrics["predicted_positive"] = sum(predicted)
        model_metrics["errors"] = [
            {
                "audit_index": row["audit_index"],
                "item_id": row["item_id"],
                "uid": row["uid"],
                "gold": g,
                "predicted": p,
                "cue_ids": by_item[row["item_id"]].get("authority_cue_ids"),
                "reason": post_by_index[int(row["audit_index"])]["reason"],
            }
            for row, g, p in zip(sealed, gold, predicted)
            if g != p
        ]
        output["models"][name] = model_metrics
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--post", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument(
        "--scores",
        action="append",
        required=True,
        help="NAME=JSONL",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    score_sets = {}
    for spec in args.scores:
        name, separator, path = spec.partition("=")
        if not separator or not name:
            raise SystemExit(f"invalid --scores value: {spec!r}")
        score_sets[name] = load_jsonl(Path(path))
    result = evaluate(
        load_jsonl(args.sealed),
        load_jsonl(args.post),
        load_jsonl(args.blind),
        score_sets,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
