#!/usr/bin/env python3
"""Benchmark text plus frozen visual features on instructional scene audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scripts.benchmark_instructional_storyboard_clip import (
        choose_threshold,
        load_examples,
        metrics,
        oof_logistic,
        sha256,
    )
except ModuleNotFoundError:
    from benchmark_instructional_storyboard_clip import (
        choose_threshold,
        load_examples,
        metrics,
        oof_logistic,
        sha256,
    )


def fit_variant(
    name: str,
    values: np.ndarray,
    labels: np.ndarray,
    train: np.ndarray,
    groups: np.ndarray,
    minimum_precision: float,
    examples: list[dict[str, Any]],
) -> dict[str, Any]:
    choices: list[tuple[float, float, np.ndarray]] = []
    for c_value in (0.001, 0.01, 0.1, 1.0):
        scores = oof_logistic(
            values[train], labels[train], groups[train], c_value
        )
        choices.append(
            (average_precision_score(labels[train], scores), c_value, scores)
        )
    train_ap, c_value, oof = max(choices, key=lambda item: item[0])
    threshold = choose_threshold(labels[train], oof, minimum_precision)
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        ),
    )
    classifier.fit(values[train], labels[train])
    scores = classifier.predict_proba(values[~train])[:, 1]
    pred = scores >= threshold
    test_rows = [row for row, keep in zip(examples, ~train) if keep]
    return {
        "name": name,
        "selected_c": c_value,
        "train_oof_average_precision": train_ap,
        "train_selected_threshold": threshold,
        "train_oof_metrics": metrics(labels[train], oof >= threshold),
        "dailymotion_transfer_metrics": metrics(labels[~train], pred),
        "dailymotion_accepted_indices": [
            row["audit_index"]
            for row, accepted in zip(test_rows, pred)
            if accepted
        ],
        "dailymotion_false_positive_indices": [
            row["audit_index"]
            for row, accepted, truth in zip(test_rows, pred, labels[~train])
            if accepted and not truth
        ],
        "dailymotion_false_negative_indices": [
            row["audit_index"]
            for row, accepted, truth in zip(test_rows, pred, labels[~train])
            if not accepted and truth
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("audit_runs"))
    parser.add_argument("--clip-cache", type=Path, required=True)
    parser.add_argument("--xclip-cache", type=Path, required=True)
    parser.add_argument("--text-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--text-model", default="sentence-transformers/all-mpnet-base-v2"
    )
    parser.add_argument("--minimum-train-precision", type=float, default=0.95)
    args = parser.parse_args()
    examples = load_examples(args.audit_root)
    hashes = [row["sheet_sha256"] for row in examples]
    clip_data = np.load(args.clip_cache)
    xclip_data = np.load(args.xclip_cache)
    if list(clip_data["sheet_sha256"]) != hashes:
        raise ValueError("CLIP cache does not match examples")
    if list(xclip_data["sheet_sha256"]) != hashes:
        raise ValueError("X-CLIP cache does not match examples")
    if args.text_cache.exists():
        cached = np.load(args.text_cache)
        text_features = cached["features"]
        if list(cached["sheet_sha256"]) != hashes:
            raise ValueError("text cache does not match examples")
    else:
        # This optional dependency pulls in a large ML stack (including
        # TensorFlow in some environments).  Keep it off the import path for
        # cached-feature runs and for callers that only use ``fit_variant``.
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(
            args.text_model,
            device="mps",
            local_files_only=True,
        )
        text_features = model.encode(
            [row["metadata_text"] for row in examples],
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).astype(np.float32)
        np.savez_compressed(
            args.text_cache,
            features=text_features,
            sheet_sha256=np.array(hashes),
        )
    clip = clip_data["features"]
    xclip = xclip_data["features"]
    labels = np.array([row["label"] for row in examples], dtype=bool)
    train = np.array([row["run"] != "v15_dailymotion" for row in examples])
    groups = np.array([row["uid"] for row in examples])
    variants = {
        "metadata_text_only": text_features,
        "clip_mean_std_plus_text": np.concatenate(
            [clip[:, :1024], text_features], axis=1
        ),
        "clip_temporal_plus_text": np.concatenate(
            [clip, text_features], axis=1
        ),
        "xclip_uniform_plus_text": np.concatenate(
            [xclip[:, :512], text_features], axis=1
        ),
        "xclip_temporal_plus_text": np.concatenate(
            [xclip, text_features], axis=1
        ),
        "clip_xclip_text": np.concatenate(
            [clip[:, :1024], xclip[:, :512], text_features], axis=1
        ),
    }
    result = {
        "kind": "instructional_multimodal_scene_development",
        "status": "posthoc_development_not_promotion",
        "examples": len(examples),
        "text_model": args.text_model,
        "minimum_train_precision": args.minimum_train_precision,
        "benchmarks": {
            name: fit_variant(
                name,
                values,
                labels,
                train,
                groups,
                args.minimum_train_precision,
                examples,
            )
            for name, values in variants.items()
        },
        "artifacts": {
            "clip_cache_sha256": sha256(args.clip_cache),
            "xclip_cache_sha256": sha256(args.xclip_cache),
            "text_cache_sha256": sha256(args.text_cache),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
