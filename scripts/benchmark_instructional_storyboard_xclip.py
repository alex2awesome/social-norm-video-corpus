#!/usr/bin/env python3
"""Benchmark frozen X-CLIP video features on manually audited storyboards."""

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
import torch
from transformers import XCLIPModel, XCLIPProcessor

try:
    from scripts.benchmark_instructional_storyboard_clip import (
        choose_threshold,
        load_examples,
        metrics,
        oof_logistic,
        sha256,
        storyboard_cells,
    )
except ModuleNotFoundError:
    from benchmark_instructional_storyboard_clip import (
        choose_threshold,
        load_examples,
        metrics,
        oof_logistic,
        sha256,
        storyboard_cells,
    )


def frame_sets(path: Path) -> list[list[np.ndarray]]:
    cells = [np.asarray(image) for image in storyboard_cells(path)]
    uniform = np.linspace(0, len(cells) - 1, 8).round().astype(int)
    windows = [
        np.linspace(start, start + 7, 8).round().astype(int)
        for start in (0, 9, 18, 28)
    ]
    return [
        [cells[index] for index in indices]
        for indices in [uniform, *windows]
    ]


def embed(
    examples: list[dict[str, Any]],
    model_name: str,
) -> np.ndarray:
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = XCLIPModel.from_pretrained(model_name, local_files_only=True).to(device)
    processor = XCLIPProcessor.from_pretrained(model_name, local_files_only=True)
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for number, row in enumerate(examples, 1):
            values: list[np.ndarray] = []
            for video in frame_sets(Path(row["sheet_path"])):
                inputs = processor(videos=video, return_tensors="pt")
                inputs = {key: value.to(device) for key, value in inputs.items()}
                feature = model.get_video_features(**inputs)
                feature = feature / feature.norm(dim=-1, keepdim=True)
                values.append(feature.cpu().numpy()[0])
            clips = np.stack(values)
            output.append(
                np.concatenate(
                    [
                        clips[0],
                        clips[1:].mean(axis=0),
                        clips[1:].std(axis=0),
                        clips[1:].max(axis=0),
                        clips[1:].reshape(-1),
                    ]
                ).astype(np.float32)
            )
            print(f"embedded {number}/{len(examples)}", flush=True)
    return np.stack(output)


def benchmark(
    examples: list[dict[str, Any]],
    features: np.ndarray,
    minimum_precision: float,
) -> dict[str, Any]:
    labels = np.array([row["label"] for row in examples], dtype=bool)
    train = np.array([row["run"] != "v15_dailymotion" for row in examples])
    test = ~train
    groups = np.array([row["uid"] for row in examples])
    variants = {
        "xclip_uniform": features[:, :512],
        "xclip_window_mean_std": features[:, 512:1536],
        "xclip_temporal_full": features,
    }
    output: dict[str, Any] = {}
    for name, values in variants.items():
        choices: list[tuple[float, float, np.ndarray]] = []
        for c_value in (0.01, 0.1, 1.0, 10.0):
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
        scores = classifier.predict_proba(values[test])[:, 1]
        pred = scores >= threshold
        test_rows = [row for row, keep in zip(examples, test) if keep]
        output[name] = {
            "selected_c": c_value,
            "train_oof_average_precision": train_ap,
            "train_selected_threshold": threshold,
            "train_oof_metrics": metrics(labels[train], oof >= threshold),
            "dailymotion_transfer_metrics": metrics(labels[test], pred),
            "dailymotion_accepted_indices": [
                row["audit_index"]
                for row, accepted in zip(test_rows, pred)
                if accepted
            ],
            "dailymotion_false_positive_indices": [
                row["audit_index"]
                for row, accepted, truth in zip(test_rows, pred, labels[test])
                if accepted and not truth
            ],
            "dailymotion_false_negative_indices": [
                row["audit_index"]
                for row, accepted, truth in zip(test_rows, pred, labels[test])
                if not accepted and truth
            ],
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("audit_runs"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/xclip-base-patch32")
    parser.add_argument("--minimum-train-precision", type=float, default=0.95)
    args = parser.parse_args()
    examples = load_examples(args.audit_root)
    if args.cache.exists():
        cached = np.load(args.cache)
        features = cached["features"]
        if list(cached["sheet_sha256"]) != [
            row["sheet_sha256"] for row in examples
        ]:
            raise ValueError("embedding cache does not match examples")
    else:
        features = embed(examples, args.model)
        np.savez_compressed(
            args.cache,
            features=features,
            sheet_sha256=np.array(
                [row["sheet_sha256"] for row in examples]
            ),
        )
    result = {
        "kind": "instructional_storyboard_xclip_development",
        "status": "posthoc_development_not_promotion",
        "model": args.model,
        "examples": len(examples),
        "minimum_train_precision": args.minimum_train_precision,
        "benchmarks": benchmark(examples, features, args.minimum_train_precision),
        "artifacts": {"cache_sha256": sha256(args.cache)},
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
