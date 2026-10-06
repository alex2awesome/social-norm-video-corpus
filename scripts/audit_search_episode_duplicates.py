#!/usr/bin/env python3
"""Flag likely duplicate search-shadow episodes for mandatory manual review.

This is an audit candidate generator, never an automatic corpus or selection
filter.  It combines duration, title-token overlap, and temporal perceptual
frame hashes so re-encodes and lightly re-edited uploads can be reviewed before
they are counted as source-disjoint replication.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


TOKEN_RE = re.compile(r"[a-z0-9]+")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dct_basis(size: int = 32) -> np.ndarray:
    positions = np.arange(size, dtype=np.float32)
    frequencies = positions[:, None]
    basis = np.cos(math.pi * (2 * positions + 1) * frequencies / (2 * size))
    basis[0] *= math.sqrt(1 / size)
    basis[1:] *= math.sqrt(2 / size)
    return basis.astype(np.float32)


_DCT32 = dct_basis(32)


def perceptual_hash(path: Path) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32)
    coefficients = (_DCT32 @ pixels @ _DCT32.T)[:8, :8]
    median = float(np.median(coefficients.ravel()[1:]))
    return (coefficients > median).ravel()


def title_tokens(title: str) -> set[str]:
    return set(TOKEN_RE.findall((title or "").lower()))


def jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def pair_metrics(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    distances = np.array([
        [int(np.count_nonzero(a != b)) for b in right["frame_hashes"]]
        for a in left["frame_hashes"]
    ], dtype=np.float32)
    aligned = np.diag(distances) if distances.shape[0] == distances.shape[1] else np.array([])
    duration_delta = abs(left["duration"] - right["duration"]) / max(
        left["duration"], right["duration"]
    )
    return {
        "relative_duration_delta": round(float(duration_delta), 6),
        "title_token_jaccard": round(jaccard(left["title_tokens"], right["title_tokens"]), 6),
        "aligned_phash_mean_hamming": round(float(np.mean(aligned)), 6) if len(aligned) else 64.0,
        "aligned_phash_median_hamming": round(float(np.median(aligned)), 6) if len(aligned) else 64.0,
        "left_best_phash_mean_hamming": round(float(np.mean(np.min(distances, axis=1))), 6),
        "right_best_phash_mean_hamming": round(float(np.mean(np.min(distances, axis=0))), 6),
        "symmetric_best_phash_mean_hamming": round(float(max(
            np.mean(np.min(distances, axis=1)), np.mean(np.min(distances, axis=0))
        )), 6),
    }


def candidate_type(
    metrics: dict[str, float],
    *,
    duration_delta_max: float,
    exact_aligned_max: float,
    episode_best_max: float,
    episode_title_jaccard_min: float,
) -> str | None:
    if metrics["relative_duration_delta"] > duration_delta_max:
        return None
    if metrics["aligned_phash_mean_hamming"] <= exact_aligned_max:
        return "near_exact_reencode"
    if (
        metrics["symmetric_best_phash_mean_hamming"] <= episode_best_max
        and metrics["title_token_jaccard"] >= episode_title_jaccard_min
    ):
        return "likely_same_episode_edit"
    return None


def load_records(label: str, manifest_path: Path) -> list[dict[str, Any]]:
    payload = json.loads(manifest_path.read_text())
    root = manifest_path.parent
    records = []
    for item in payload.get("records") or []:
        if item.get("artifact_status") != "rendered":
            continue
        duration = float(item.get("probed_duration") or 0)
        frames = item.get("frames") or []
        if duration <= 0 or not frames:
            continue
        frame_paths = [root / frame["path"] for frame in frames]
        missing = [str(path) for path in frame_paths if not path.is_file()]
        if missing:
            raise SystemExit(f"missing frames for {item.get('uid')}: {missing[:2]}")
        records.append({
            "run": label,
            "uid": item["uid"],
            "title": item.get("title") or "",
            "duration": duration,
            "manifest": str(manifest_path),
            "sheet_path": item.get("sheet_path"),
            "sheet_sha256": item.get("sheet_sha256"),
            "media_sha256": item.get("media_sha256"),
            "title_tokens": title_tokens(item.get("title") or ""),
            "frame_hashes": [perceptual_hash(path) for path in frame_paths],
        })
    return records


def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key) for key in (
        "run", "uid", "title", "duration", "manifest", "sheet_path",
        "sheet_sha256", "media_sha256",
    )}


def parse_manifest(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("manifest must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label:
        raise argparse.ArgumentTypeError("manifest label cannot be empty")
    return label, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=parse_manifest)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration-delta-max", type=float, default=0.005)
    parser.add_argument("--exact-aligned-max", type=float, default=4.0)
    parser.add_argument("--episode-best-max", type=float, default=20.0)
    parser.add_argument("--episode-title-jaccard-min", type=float, default=0.30)
    args = parser.parse_args()

    if len({label for label, _path in args.manifest}) != len(args.manifest):
        raise SystemExit("manifest labels must be unique")
    records = []
    manifests = []
    for label, path in args.manifest:
        if not path.is_file():
            raise SystemExit(f"manifest not found: {path}")
        manifests.append({"label": label, "path": str(path), "sha256": file_sha256(path)})
        records.extend(load_records(label, path))

    pairs = []
    for left, right in itertools.combinations(records, 2):
        metrics = pair_metrics(left, right)
        kind = candidate_type(
            metrics,
            duration_delta_max=args.duration_delta_max,
            exact_aligned_max=args.exact_aligned_max,
            episode_best_max=args.episode_best_max,
            episode_title_jaccard_min=args.episode_title_jaccard_min,
        )
        if kind is None:
            continue
        pairs.append({
            "candidate_type": kind,
            "left": public_record(left),
            "right": public_record(right),
            "metrics": metrics,
            "manual_decision": None,
            "manual_notes": None,
        })
    pairs.sort(key=lambda row: (
        row["metrics"]["symmetric_best_phash_mean_hamming"],
        row["left"]["run"], row["left"]["uid"], row["right"]["run"], row["right"]["uid"],
    ))
    result = {
        "kind": "search_shadow_episode_duplicate_candidates_v1",
        "automatic_action": "none_manual_review_required",
        "manifests": manifests,
        "thresholds": {
            "relative_duration_delta_max": args.duration_delta_max,
            "near_exact_aligned_phash_mean_hamming_max": args.exact_aligned_max,
            "same_episode_symmetric_best_phash_mean_hamming_max": args.episode_best_max,
            "same_episode_title_token_jaccard_min": args.episode_title_jaccard_min,
        },
        "rendered_sources_compared": len(records),
        "all_pairs_compared": len(records) * (len(records) - 1) // 2,
        "candidate_pairs": len(pairs),
        "pairs": pairs,
    }
    result["pairs_sha256"] = hashlib.sha256(
        json.dumps(pairs, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    review = args.out.with_suffix(".manual_review.tsv")
    with review.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow([
            "pair_index", "candidate_type", "left_run", "left_uid", "right_run", "right_uid",
            "manual_decision", "same_underlying_episode", "independent_replication", "manual_notes",
        ])
        for index, pair in enumerate(pairs):
            writer.writerow([
                index, pair["candidate_type"], pair["left"]["run"], pair["left"]["uid"],
                pair["right"]["run"], pair["right"]["uid"], "", "", "", "",
            ])
    print(json.dumps({"sources": len(records), "pairs": len(pairs), "review": str(review)}))


if __name__ == "__main__":
    main()
