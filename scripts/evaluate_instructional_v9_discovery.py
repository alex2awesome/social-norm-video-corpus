#!/usr/bin/env python3
"""Evaluate frozen V9 shadow records against the 90-clip manual ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_success(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("result") is not None and row.get("error") is None:
            records[row["item_id"]] = row
    return records


def wilson_95(successes: int, total: int) -> list[float] | None:
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / denominator
    )
    return [center - radius, center + radius]


def metric(
    rows: list[dict[str, Any]],
    selected: Callable[[dict[str, Any]], bool],
    target: str,
) -> dict[str, Any]:
    chosen = [row for row in rows if selected(row)]
    positives = sum(bool(row["manual"][target]) for row in chosen)
    total_target = sum(bool(row["manual"][target]) for row in rows)
    return {
        "selected": len(chosen),
        "positive": positives,
        "false_positive": len(chosen) - positives,
        "precision": positives / len(chosen) if chosen else None,
        "precision_wilson_95": wilson_95(positives, len(chosen)),
        "recall": positives / total_target if total_target else None,
        "selected_audit_indices": sorted(
            row["manual"]["audit_index"] for row in chosen
        ),
        "false_positive_audit_indices": sorted(
            row["manual"]["audit_index"]
            for row in chosen
            if not row["manual"][target]
        ),
        "false_negative_audit_indices": sorted(
            row["manual"]["audit_index"]
            for row in rows
            if row["manual"][target] and not selected(row)
        ),
    }


def original_violation_usable(row: dict[str, Any], alignment: str) -> bool:
    """Frozen discovery rule using upstream polarity, not model-restated polarity."""
    return (
        row["manifest"].get("polarity") == "violation"
        and row["alignments"][alignment]["usable_after_relabel"] == "yes"
    )


def evaluate(
    manual_path: Path,
    manifest_path: Path,
    qwen_video_path: Path,
    glm_video_path: Path,
    storyboard_path: Path,
    alignments: list[tuple[str, Path]],
) -> dict[str, Any]:
    manual = {row["item_id"]: row for row in read_jsonl(manual_path)}
    manifest = {row["item_id"]: row for row in read_jsonl(manifest_path)}
    qwen_video = latest_success(qwen_video_path)
    glm_video = latest_success(glm_video_path)
    storyboard = latest_success(storyboard_path)
    alignment_records = {
        name: latest_success(path) for name, path in alignments
    }
    expected = set(manual)
    sources = {
        "manifest": manifest,
        "qwen_video": qwen_video,
        "glm_video": glm_video,
        "storyboard": storyboard,
        **{f"alignment:{name}": rows for name, rows in alignment_records.items()},
    }
    for name, records in sources.items():
        if set(records) != expected:
            raise ValueError(
                f"{name}: expected {len(expected)} item IDs, found {len(records)}"
            )
    if len(expected) != 90:
        raise ValueError(f"expected 90 manual items, found {len(expected)}")

    rows = []
    for item_id in sorted(expected, key=lambda key: manual[key]["audit_index"]):
        rows.append(
            {
                "item_id": item_id,
                "manual": manual[item_id],
                "manifest": manifest[item_id],
                "qwen_video": qwen_video[item_id]["result"],
                "glm_video": glm_video[item_id]["result"],
                "storyboard": storyboard[item_id]["result"],
                "alignments": {
                    name: records[item_id]["result"]
                    for name, records in alignment_records.items()
                },
            }
        )

    q_yes = lambda row: row["qwen_video"]["observable_event"] == "yes"
    g_yes = lambda row: row["glm_video"]["observable_event"] == "yes"
    s_yes = lambda row: row["storyboard"]["observable_event"] == "yes"
    rules: dict[str, Callable[[dict[str, Any]], bool]] = {
        "qwen_video_v9a_event": q_yes,
        "glm_video_v9a_event": g_yes,
        "storyboard_v9a_event": s_yes,
        "qwen_and_glm_video_event": lambda row: q_yes(row) and g_yes(row),
        "all_three_visual_event": lambda row: q_yes(row)
        and g_yes(row)
        and s_yes(row),
    }
    for name in alignment_records:
        rules[f"{name}_exact"] = (
            lambda row, key=name: row["alignments"][key][
                "exact_weak_label_usable"
            ]
            == "yes"
        )
        rules[f"{name}_usable_after_relabel"] = (
            lambda row, key=name: row["alignments"][key][
                "usable_after_relabel"
            ]
            == "yes"
        )
        rules[f"{name}_supported_relation_plus_glm_v8"] = (
            lambda row, key=name: row["alignments"][key][
                "proposed_norm_relation"
            ]
            in {"exact", "broader_but_supported"}
            and row["manifest"]["glm_v8_result"]["usable_demo_after_relabel"]
            == "yes"
        )
        rules[f"{name}_violation_band"] = (
            lambda row, key=name: row["alignments"][key][
                "usable_after_relabel"
            ]
            == "yes"
            and row["alignments"][key]["visible_event_polarity"] == "violation"
        )
        rules[f"{name}_original_violation_usable"] = (
            lambda row, key=name: original_violation_usable(row, key)
        )

    targets = (
        "any_visual_demo",
        "exact_weak_label_usable",
        "usable_after_relabel",
    )
    metrics = {
        rule_name: {
            target: metric(rows, rule, target)
            for target in targets
        }
        for rule_name, rule in rules.items()
    }
    return {
        "kind": "instructional_v9_discovery_evaluation",
        "status": "discovery_only_not_source_disjoint",
        "n": len(rows),
        "input_hashes": {
            "manual": sha256(manual_path),
            "manifest": sha256(manifest_path),
            "qwen_video": sha256(qwen_video_path),
            "glm_video": sha256(glm_video_path),
            "storyboard": sha256(storyboard_path),
            **{f"alignment:{name}": sha256(path) for name, path in alignments},
        },
        "manual_counts": dict(
            sorted(
                Counter(
                    row["manual"]["semantic_status"] for row in rows
                ).items()
            )
        ),
        "metrics": metrics,
        "decision": (
            "No rule may be promoted from this reused 90-clip discovery set. "
            "Any candidate must be frozen and evaluated on a new UID-disjoint "
            "manual holdout."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen-video", type=Path, required=True)
    parser.add_argument("--glm-video", type=Path, required=True)
    parser.add_argument("--storyboard", type=Path, required=True)
    parser.add_argument(
        "--alignment",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Repeat for each V9B alignment model.",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    alignments = []
    for item in args.alignment:
        if "=" not in item:
            raise ValueError("--alignment must be NAME=PATH")
        name, path = item.split("=", 1)
        alignments.append((name, Path(path)))
    summary = evaluate(
        args.manual,
        args.manifest,
        args.qwen_video,
        args.glm_video,
        args.storyboard,
        alignments,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
