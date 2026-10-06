#!/usr/bin/env python3
"""Leakage probes: how much label signal lives in NON-video channels?

Trains tiny hashed bag-of-words logistic probes on the R1 manifest's train
split and evaluates on the test split, using only text channels a video
model should NOT need: (a) the source transcript excerpt, (b) source
metadata (title + description).  Probe AUROC materially above 0.5 means the
task can be partially solved without watching the video — reported beside
every headline metric so shortcuts are measured, not discovered late.
Pure numpy; no GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from experiments.eval_metrics_v1 import auroc, sliced_auroc

PROBE_VERSION = "leakage_probes_v1"
DIM = 4096
TOKEN = re.compile(r"[a-zA-Z']{2,}")


def featurize(text: str) -> np.ndarray:
    vector = np.zeros(DIM, dtype=np.float32)
    for token in TOKEN.findall(text.lower())[:400]:
        index = int(hashlib.md5(token.encode()).hexdigest()[:8], 16) % DIM
        vector[index] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def train_logistic(
    features: np.ndarray, labels: np.ndarray,
    *, epochs: int = 60, lr: float = 0.5, l2: float = 1e-4,
) -> np.ndarray:
    weights = np.zeros(features.shape[1] + 1, dtype=np.float32)
    padded = np.hstack([features, np.ones((len(features), 1), dtype=np.float32)])
    for _ in range(epochs):
        logits = padded @ weights
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        gradient = padded.T @ (probabilities - labels) / len(labels) + l2 * weights
        weights -= lr * gradient
    return weights


def predict(weights: np.ndarray, features: np.ndarray) -> np.ndarray:
    padded = np.hstack([features, np.ones((len(features), 1), dtype=np.float32)])
    return 1.0 / (1.0 + np.exp(-(padded @ weights)))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_channels(root: Path, manifest: Path, metadata_path: Path) -> list[dict[str, Any]]:
    metadata = {r["uid"]: r for r in iter_jsonl(metadata_path)}
    transcript_cache: dict[str, str] = {}
    rows = []
    for item in iter_jsonl(manifest):
        uid = item["uid"]
        if uid not in transcript_cache:
            path = root / "data" / "transcripts" / f"{uid}.json"
            text = ""
            if path.is_file():
                try:
                    segments = json.loads(path.read_text()).get("segments") or []
                    text = " ".join(str(s.get("text") or "") for s in segments)[:1500]
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
            transcript_cache[uid] = text
        source_meta = metadata.get(uid) or {}
        rows.append({
            "item_id": item["item_id"], "split": item["split"],
            "label": item["label_reaction_present"],
            "platform": item["covariates"].get("platform"),
            "kind": item["kind"],
            "transcript_text": transcript_cache[uid],
            "metadata_text": " ".join(
                str(source_meta.get(k) or "") for k in ("title", "description")
            )[:1500],
        })
    return rows


def run_probe(rows: list[dict[str, Any]], channel: str) -> dict[str, Any]:
    train = [r for r in rows if r["split"] == "train" and r[channel]]
    test = [r for r in rows if r["split"] == "test" and r[channel]]
    if len(train) < 100 or len(test) < 50:
        return {"channel": channel, "error": "insufficient covered rows",
                "train": len(train), "test": len(test)}
    weights = train_logistic(
        np.stack([featurize(r[channel]) for r in train]),
        np.array([r["label"] for r in train], dtype=np.float32),
    )
    scores = predict(weights, np.stack([featurize(r[channel]) for r in test]))
    for row, score in zip(test, scores):
        row[f"score_{channel}"] = float(score)
    return {
        "channel": channel, "train": len(train), "test": len(test),
        "auroc": auroc([float(s) for s in scores], [r["label"] for r in test]),
        "auroc_by_platform": sliced_auroc(test, f"score_{channel}", "label", "platform"),
        "coverage_test": len(test) / max(1, sum(1 for r in rows if r["split"] == "test")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    rows = load_channels(args.root, args.manifest, args.metadata)
    report = {
        "probe_version": PROBE_VERSION,
        "items": len(rows),
        "probes": [run_probe(rows, channel)
                   for channel in ("transcript_text", "metadata_text")],
        "interpretation": "AUROC >> 0.5 means the reaction label is partially "
                          "recoverable without video; compare against VLM AUROC",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({p["channel"]: p.get("auroc") for p in report["probes"]},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
