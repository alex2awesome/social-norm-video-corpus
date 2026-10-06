#!/usr/bin/env python3
"""Fit V19 multimodal shadow features on V15+V17 and transfer to V18.

This benchmark is read-only. It evaluates low-level, pose, CLIP, and X-CLIP
features against frozen manual labels. It does not create keep/reject authority.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.benchmark_instructional_storyboard_clip import sha256
    from scripts.benchmark_instructional_v19_cheap_features import (
        read_jsonl,
        score_transfer,
    )
    from scripts.evaluate_full_corpus_multimodal_features import (
        flatten,
        last_successful_rows,
    )
except ModuleNotFoundError:
    from benchmark_instructional_storyboard_clip import sha256
    from benchmark_instructional_v19_cheap_features import (
        read_jsonl,
        score_transfer,
    )
    from evaluate_full_corpus_multimodal_features import (
        flatten,
        last_successful_rows,
    )


PREFIXES = {
    "low_level": ("low.",),
    "pose": ("pose.",),
    "clip": ("clip.",),
    "xclip": ("xclip.",),
    "low_plus_pose": ("low.", "pose."),
    "clip_plus_xclip": ("clip.", "xclip."),
    "all_visual": ("low.", "pose.", "clip.", "xclip."),
}


def feature_groups(names: set[str]) -> dict[str, list[str]]:
    atomic = {
        group: sorted(
            name
            for name in names
            if any(name.startswith(prefix) for prefix in prefixes)
        )
        for group, prefixes in PREFIXES.items()
        if group in {"low_level", "pose", "clip", "xclip"}
    }
    groups = {group: values for group, values in atomic.items() if values}
    if atomic["low_level"] and atomic["pose"]:
        groups["low_plus_pose"] = sorted(
            atomic["low_level"] + atomic["pose"]
        )
    if atomic["clip"] and atomic["xclip"]:
        groups["clip_plus_xclip"] = sorted(
            atomic["clip"] + atomic["xclip"]
        )
    groups["all_visual"] = sorted(
        value
        for values in groups.values()
        for value in values
    )
    groups["all_visual"] = sorted(set(groups["all_visual"]))
    return {group: values for group, values in groups.items() if values}


def benchmark(
    manifest_rows: list[dict[str, Any]],
    score_rows: dict[str, dict[str, Any]],
    minimum_precision: float,
) -> dict[str, Any]:
    if len(manifest_rows) != 168:
        raise ValueError(f"expected 168 benchmark rows, got {len(manifest_rows)}")
    wanted = {str(row["item_id"]) for row in manifest_rows}
    missing = wanted - set(score_rows)
    if missing:
        raise ValueError(f"missing {len(missing)} successful score rows")

    flat = {
        item_id: flatten(score_rows[item_id])
        for item_id in wanted
    }
    names = set.intersection(*(set(values) for values in flat.values()))
    groups = feature_groups(names)
    atomic = [
        name for name in ("low_level", "pose", "clip", "xclip")
        if name in groups
    ]
    if not atomic:
        raise ValueError("no common visual feature groups are available")

    train = np.asarray(
        [row["audit_cohort"] in {"v15", "v17"} for row in manifest_rows]
    )
    uids = np.asarray([row["uid"] for row in manifest_rows])
    targets = {
        "visual": np.asarray(
            [bool(row["gold_scene_visible"]) for row in manifest_rows]
        ),
        "usable": np.asarray(
            [bool(row["gold_usable"]) for row in manifest_rows]
        ),
        "exact": np.asarray(
            [bool(row["gold_exact_social_norm"]) for row in manifest_rows]
        ),
    }
    results: dict[str, Any] = {}
    for group, feature_names in groups.items():
        values = np.asarray(
            [
                [flat[str(row["item_id"])][name] for name in feature_names]
                for row in manifest_rows
            ],
            dtype=np.float64,
        )
        results[group] = {
            target: score_transfer(
                values,
                labels,
                train,
                uids,
                minimum_precision,
            )
            for target, labels in targets.items()
        }
    return {
        "kind": "instructional_v19_multimodal_feature_benchmark",
        "status": "posthoc_development_not_promotion",
        "policy": "read_only_shadow_features_no_keep_or_reject_authority",
        "protocol": "fit_v15_v17_test_v18_source_disjoint",
        "coverage": {
            "total": len(manifest_rows),
            "train_v15_v17": int(train.sum()),
            "test_v18": int((~train).sum()),
        },
        "minimum_train_precision": minimum_precision,
        "available_atomic_feature_groups": atomic,
        "feature_groups": groups,
        "model_results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--scores",
        type=Path,
        action="append",
        required=True,
        help="Append-only feature JSONL; repeat to merge feature sections/retries",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-train-precision", type=float, default=0.90)
    args = parser.parse_args()
    report = benchmark(
        read_jsonl(args.manifest),
        last_successful_rows(args.scores),
        args.minimum_train_precision,
    )
    report["artifact_sha256"] = {
        "manifest": sha256(args.manifest),
        "scores": [
            {"path": str(path), "sha256": sha256(path)}
            for path in args.scores
        ],
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
