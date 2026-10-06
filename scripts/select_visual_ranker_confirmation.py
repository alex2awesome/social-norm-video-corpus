#!/usr/bin/env python3
"""Freeze source-disjoint top/middle/bottom visual-ranker confirmation sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import clone

if __package__:
    from scripts.evaluate_full_corpus_audit_features import load_gold
    from scripts.evaluate_full_corpus_multimodal_features import (
        estimators,
        flatten,
        last_successful_rows,
        matrix,
    )
else:
    from evaluate_full_corpus_audit_features import load_gold
    from evaluate_full_corpus_multimodal_features import (
        estimators,
        flatten,
        last_successful_rows,
        matrix,
    )


FEATURE_PREFIXES = ("low.", "pose.", "clip.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def last_candidate_rows(paths: list[Path]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            if (
                row.get("error") is None
                and row.get("item_id")
                and isinstance(row.get("clip_scores"), dict)
                and isinstance(row.get("keypoints"), dict)
            ):
                result[row["item_id"]] = row
    return result


def choose_score_bands(
    rows: list[dict[str, Any]],
    per_band: int,
    seed: str,
) -> list[dict[str, Any]]:
    scores = np.asarray([float(row["visual_score"]) for row in rows])
    low, middle_left, middle_right, high = np.quantile(
        scores, [0.2, 0.4, 0.6, 0.8]
    )
    cells = {
        "bottom_quintile": [
            row for row in rows if float(row["visual_score"]) <= low
        ],
        "middle_quintile": [
            row
            for row in rows
            if middle_left
            <= float(row["visual_score"])
            <= middle_right
        ],
        "top_quintile": [
            row for row in rows if float(row["visual_score"]) >= high
        ],
    }
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    used_uids: set[str] = set()
    for band in ("bottom_quintile", "middle_quintile", "top_quintile"):
        candidates = list(cells[band])
        rng.shuffle(candidates)
        chosen = []
        for row in candidates:
            if row["uid"] in used_uids:
                continue
            chosen.append({**row, "score_band": band})
            used_uids.add(row["uid"])
            if len(chosen) == per_band:
                break
        if len(chosen) != per_band:
            raise ValueError(
                f"{band}: wanted {per_band}, found {len(chosen)}"
            )
        selected.extend(chosen)
    rng.shuffle(selected)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", required=True, type=Path)
    parser.add_argument(
        "--train-features", required=True, type=Path, action="append"
    )
    parser.add_argument(
        "--candidate-features", required=True, type=Path, action="append"
    )
    parser.add_argument(
        "--pillar", required=True, choices=("instructional", "witnessed")
    )
    parser.add_argument("--exclude-uids", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--per-band", type=int, default=10)
    parser.add_argument("--seed", default="visual-ranker-confirmation-v1")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    training = load_gold(args.audit_root, "mechanism", args.pillar)
    train_ids = sorted(training)
    merged = last_successful_rows(
        [*args.train_features, *args.candidate_features]
    )
    flat = {item_id: flatten(row) for item_id, row in merged.items()}
    candidate_rows = last_candidate_rows(args.candidate_features)
    excluded = {
        line.strip()
        for line in args.exclude_uids.read_text().splitlines()
        if line.strip()
    }
    eligible_ids = sorted(
        item_id
        for item_id, row in candidate_rows.items()
        if row["pillar"] == args.pillar
        and row["uid"] not in excluded
        and item_id in flat
    )
    missing_training = sorted(set(train_ids) - set(flat))
    if missing_training:
        raise ValueError(f"missing training features: {missing_training}")
    feature_names = sorted(
        {
            name
            for item_id in train_ids + eligible_ids
            for name in flat[item_id]
            if name.startswith(FEATURE_PREFIXES)
        }
    )
    x_train = matrix(train_ids, flat, feature_names)
    x_candidates = matrix(eligible_ids, flat, feature_names)
    y_train = np.asarray(
        [int(training[item_id]["visual"]) for item_id in train_ids],
        dtype=int,
    )
    model = clone(estimators()["logistic"]).fit(x_train, y_train)
    probabilities = model.predict_proba(x_candidates)[:, 1]
    scored = []
    for item_id, score in zip(eligible_ids, probabilities):
        source = candidate_rows[item_id]
        scored.append(
            {
                "item_id": item_id,
                "uid": source["uid"],
                "pillar": source["pillar"],
                "clip": source["clip"],
                "visual_score": float(score),
            }
        )
    selected = choose_score_bands(scored, args.per_band, args.seed)
    args.out.mkdir(parents=True)
    blind = []
    sealed = []
    for audit_index, row in enumerate(selected):
        blind.append(
            {
                "audit_index": audit_index,
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "clip": row["clip"],
            }
        )
        sealed.append(
            {
                **row,
                "audit_index": audit_index,
            }
        )
    blind_path = args.out / "blind_manifest.jsonl"
    sealed_path = args.out / "sealed_selection.jsonl"
    blind_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind),
        encoding="utf-8",
    )
    sealed_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sealed),
        encoding="utf-8",
    )
    provenance = {
        "schema_version": 1,
        "kind": "visual_ranker_score_band_confirmation",
        "pillar": args.pillar,
        "protocol": "fit_mechanism_60_select_new_source_disjoint_score_bands",
        "policy": "blind_manual_confirmation_shadow_only",
        "seed": args.seed,
        "eligible_candidates": len(scored),
        "per_band": args.per_band,
        "selected": len(selected),
        "excluded_uids": len(excluded),
        "feature_names": feature_names,
        "training_positives": int(y_train.sum()),
        "training_items": len(train_ids),
        "train_feature_sha256": {
            str(path): sha256(path) for path in args.train_features
        },
        "candidate_feature_sha256": {
            str(path): sha256(path) for path in args.candidate_features
        },
        "blind_manifest_sha256": sha256(blind_path),
        "sealed_selection_sha256": sha256(sealed_path),
    }
    (args.out / "preregistration.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
