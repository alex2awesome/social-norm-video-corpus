#!/usr/bin/env python3
"""Benchmark frozen CLIP storyboard features against manual scene audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from transformers import CLIPModel, CLIPProcessor


RUN_SPECS = {
    "v12_youtube": {
        "dir": "20260728_instructional_v12_holdout_v1",
        "ledger": "blind_visual_ledger.tsv",
    },
    "v13_youtube": {
        "dir": "20260728_instructional_v13_simple_youtube_visual_holdout_v1",
        "ledger": "blind_visual_ledger.tsv",
    },
    "v14_youtube": {
        "dir": "20260728_instructional_v14_response_holdout_v1",
        "ledger": "blind_visual_ledger.tsv",
    },
    "v15_dailymotion": {
        "dir": "20260728_instructional_v15_utterance_anchor_prospective_v1",
        "ledger": "manual_visual_ledger_blind.tsv",
    },
}
POSITIVE_TEXTS = [
    "people acting out a specific social conflict",
    "one person insulting excluding threatening or deceiving another person",
    "a concrete etiquette violation with another person reacting",
    "a person sharing helping apologizing or taking turns with someone",
    "a public nuisance or reckless act visibly affecting other people",
]
NEGATIVE_TEXTS = [
    "a talking head presenter lecture interview podcast or panel discussion",
    "a news report or stock footage montage",
    "a medical health diet safety or technical procedure",
    "an ordinary conversation with no visible social conflict",
    "a slide diagram title card or on screen text",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"y", "yes", "true", "1"}


def resolve_images(run_dir: Path, manifest: list[dict[str, Any]]) -> dict[str, Path]:
    wanted = {str(row["sheet_sha256"]) for row in manifest}
    found: dict[str, Path] = {}
    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            digest = sha256(path)
            if digest in wanted:
                if digest in found and found[digest] != path:
                    continue
                found[digest] = path
    missing = wanted - set(found)
    if missing:
        raise FileNotFoundError(
            f"{run_dir}: missing {len(missing)} storyboard hashes"
        )
    return found


def load_examples(audit_root: Path) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for run_name, spec in RUN_SPECS.items():
        run_dir = audit_root / spec["dir"]
        manifest = read_jsonl(run_dir / "blind_manifest.jsonl")
        by_index = {int(row["audit_index"]): row for row in manifest}
        semantic = {
            int(row["audit_index"]): row
            for row in read_jsonl(run_dir / "audit_selection_semantic.jsonl")
        }
        with (run_dir / spec["ledger"]).open() as handle:
            labels = list(csv.DictReader(handle, delimiter="\t"))
        images = resolve_images(run_dir, manifest)
        for label in labels:
            index = int(label["audit_index"])
            if truthy(label.get("protocol_exception", "false")):
                continue
            value = str(label["visual_demo"]).strip().upper()
            if value not in {"Y", "N"}:
                raise ValueError(
                    f"{run_name} audit_index {index}: unsupported label {value}"
                )
            blind = by_index[index]
            source = semantic[index]
            metadata_text = " ".join(
                str(source.get(key) or "")
                for key in (
                    "title",
                    "category",
                    "norm",
                    "start_quote",
                    "end_quote",
                    "explanation",
                )
            ).strip()
            examples.append(
                {
                    "run": run_name,
                    "audit_index": index,
                    "candidate_id": blind.get(
                        "candidate_id", f"{run_name}-{index:04d}"
                    ),
                    "uid": source["uid"],
                    "label": value == "Y",
                    "metadata_text": metadata_text,
                    "sheet_path": str(images[str(blind["sheet_sha256"])]),
                    "sheet_sha256": str(blind["sheet_sha256"]),
                }
            )
    return examples


def storyboard_cells(path: Path, rows: int = 9, cols: int = 4) -> list[Image.Image]:
    image = Image.open(path).convert("RGB")
    width, height = image.size
    return [
        image.crop(
            (
                round(col * width / cols),
                round(row * height / rows),
                round((col + 1) * width / cols),
                round((row + 1) * height / rows),
            )
        )
        for row in range(rows)
        for col in range(cols)
    ]


def embed(
    examples: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = CLIPModel.from_pretrained(model_name, local_files_only=True)
    model.to(device)
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)
    model.eval()
    all_features: list[np.ndarray] = []
    with torch.inference_mode():
        for number, example in enumerate(examples, 1):
            cells = storyboard_cells(Path(example["sheet_path"]))
            chunks: list[np.ndarray] = []
            for start in range(0, len(cells), batch_size):
                inputs = processor(
                    images=cells[start : start + batch_size],
                    return_tensors="pt",
                )
                inputs = {key: value.to(device) for key, value in inputs.items()}
                values = model.get_image_features(**inputs)
                values = values / values.norm(dim=-1, keepdim=True)
                chunks.append(values.cpu().numpy())
            frames = np.concatenate(chunks)
            segments = np.stack(
                [part.mean(axis=0) for part in np.array_split(frames, 4)]
            )
            feature = np.concatenate(
                [
                    frames.mean(axis=0),
                    frames.std(axis=0),
                    frames.max(axis=0),
                    segments.reshape(-1),
                ]
            )
            all_features.append(feature.astype(np.float32))
            print(f"embedded {number}/{len(examples)}", flush=True)

        text_inputs = processor(
            text=POSITIVE_TEXTS + NEGATIVE_TEXTS,
            padding=True,
            return_tensors="pt",
        )
        text_inputs = {
            key: value.to(device) for key, value in text_inputs.items()
        }
        text = model.get_text_features(**text_inputs)
        text = text / text.norm(dim=-1, keepdim=True)
        text_features = text.cpu().numpy()
    return np.stack(all_features), text_features


def metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    fn = int(np.sum(~pred & y))
    tn = int(np.sum(~pred & ~y))
    return {
        "n": int(len(y)),
        "accepted": int(np.sum(pred)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "specificity": tn / (tn + fp) if tn + fp else None,
    }


def choose_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    minimum_precision: float,
) -> float:
    choices: list[tuple[float, float, float]] = []
    for threshold in np.unique(np.concatenate([scores, [1.0 + scores.max()]])):
        pred = scores >= threshold
        result = metrics(y, pred)
        precision = result["precision"]
        if precision is not None and precision >= minimum_precision:
            choices.append((float(result["recall"]), -float(threshold), float(threshold)))
    return max(choices)[2] if choices else float(1.0 + scores.max())


def oof_logistic(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    c_value: float,
) -> np.ndarray:
    splits = min(5, len(np.unique(groups)))
    output = np.zeros(len(labels), dtype=np.float64)
    for train, valid in GroupKFold(n_splits=splits).split(
        features, labels, groups
    ):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=5000,
                random_state=0,
            ),
        )
        classifier.fit(features[train], labels[train])
        output[valid] = classifier.predict_proba(features[valid])[:, 1]
    return output


def benchmark(
    examples: list[dict[str, Any]],
    features: np.ndarray,
    text_features: np.ndarray,
    minimum_precision: float,
) -> dict[str, Any]:
    labels = np.array([row["label"] for row in examples], dtype=bool)
    train = np.array([row["run"] != "v15_dailymotion" for row in examples])
    test = ~train
    groups = np.array([row["uid"] for row in examples])
    representations = {
        "clip_mean": features[:, :512],
        "clip_mean_std": features[:, :1024],
        "clip_temporal_full": features,
    }
    output: dict[str, Any] = {}
    for name, values in representations.items():
        candidates: list[tuple[float, float, np.ndarray]] = []
        for c_value in (0.01, 0.1, 1.0, 10.0):
            scores = oof_logistic(
                values[train], labels[train], groups[train], c_value
            )
            candidates.append(
                (average_precision_score(labels[train], scores), c_value, scores)
            )
        train_ap, c_value, oof = max(candidates, key=lambda item: item[0])
        threshold = choose_threshold(
            labels[train], oof, minimum_precision
        )
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
        test_scores = classifier.predict_proba(values[test])[:, 1]
        test_pred = test_scores >= threshold
        test_rows = [row for row, keep in zip(examples, test) if keep]
        output[name] = {
            "selected_c": c_value,
            "train_oof_average_precision": train_ap,
            "train_selected_threshold": threshold,
            "train_oof_metrics": metrics(labels[train], oof >= threshold),
            "dailymotion_transfer_metrics": metrics(labels[test], test_pred),
            "dailymotion_accepted_indices": [
                row["audit_index"]
                for row, accepted in zip(test_rows, test_pred)
                if accepted
            ],
            "dailymotion_false_positive_indices": [
                row["audit_index"]
                for row, accepted, truth in zip(
                    test_rows, test_pred, labels[test]
                )
                if accepted and not truth
            ],
            "dailymotion_false_negative_indices": [
                row["audit_index"]
                for row, accepted, truth in zip(
                    test_rows, test_pred, labels[test]
                )
                if not accepted and truth
            ],
        }

    image_mean = features[:, :512]
    positive = image_mean @ text_features[: len(POSITIVE_TEXTS)].T
    negative = image_mean @ text_features[len(POSITIVE_TEXTS) :].T
    zero_scores = positive.max(axis=1) - negative.max(axis=1)
    threshold = choose_threshold(
        labels[train], zero_scores[train], minimum_precision
    )
    zero_pred = zero_scores[test] >= threshold
    test_rows = [row for row, keep in zip(examples, test) if keep]
    output["clip_zero_shot"] = {
        "train_selected_threshold": threshold,
        "train_metrics": metrics(
            labels[train], zero_scores[train] >= threshold
        ),
        "dailymotion_transfer_metrics": metrics(labels[test], zero_pred),
        "dailymotion_false_positive_indices": [
            row["audit_index"]
            for row, accepted, truth in zip(test_rows, zero_pred, labels[test])
            if accepted and not truth
        ],
        "dailymotion_false_negative_indices": [
            row["audit_index"]
            for row, accepted, truth in zip(test_rows, zero_pred, labels[test])
            if not accepted and truth
        ],
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("audit_runs"))
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model", default="openai/clip-vit-base-patch32"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-train-precision", type=float, default=0.95)
    args = parser.parse_args()
    examples = load_examples(args.audit_root)
    if args.cache.exists():
        cached = np.load(args.cache)
        features = cached["features"]
        text_features = cached["text_features"]
        hashes = list(cached["sheet_sha256"])
        if hashes != [row["sheet_sha256"] for row in examples]:
            raise ValueError("embedding cache does not match examples")
    else:
        features, text_features = embed(
            examples, args.model, args.batch_size
        )
        np.savez_compressed(
            args.cache,
            features=features,
            text_features=text_features,
            sheet_sha256=np.array(
                [row["sheet_sha256"] for row in examples]
            ),
        )
    result = {
        "kind": "instructional_storyboard_clip_development",
        "status": "posthoc_development_not_promotion",
        "model": args.model,
        "examples": len(examples),
        "train_examples": sum(
            row["run"] != "v15_dailymotion" for row in examples
        ),
        "dailymotion_transfer_examples": sum(
            row["run"] == "v15_dailymotion" for row in examples
        ),
        "run_counts": {
            name: sum(row["run"] == name for row in examples)
            for name in RUN_SPECS
        },
        "minimum_train_precision": args.minimum_train_precision,
        "benchmarks": benchmark(
            examples, features, text_features, args.minimum_train_precision
        ),
        "artifacts": {
            "cache_sha256": sha256(args.cache),
        },
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
