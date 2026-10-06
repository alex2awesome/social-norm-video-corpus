#!/usr/bin/env python3
"""Partition an instructional rescore queue by strict conditioned-v5 votes.

This is a shadow-data operation only. It never changes corpus metadata or
deletes clips. With one VLM ledger it emits that model's strict-pass queue;
with two ledgers it also emits their strict intersection and reports the
disagreement bands for later manual audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


STRICT_YES_FIELDS = (
    "social_norm_domain",
    "behavior_occurs_in_scene",
    "affected_party_or_shared_setting_visible",
    "situated_interaction_complete",
    "usable_demo_after_relabel",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def latest_successes(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Keep the latest valid row from an append-only inference ledger."""
    successful: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("result") is not None and not row.get("error"):
            successful[row["item_id"]] = row
    return successful


def strict_v5_pass(result: dict[str, Any]) -> bool:
    return all(result.get(field) == "yes" for field in STRICT_YES_FIELDS)


def _annotate(
    row: dict[str, Any],
    primary: dict[str, Any],
    secondary: dict[str, Any] | None,
) -> dict[str, Any]:
    record = dict(row)
    record["qwen_v5_conditioned_result"] = primary["result"]
    if secondary is not None:
        record["glm_v5_conditioned_result"] = secondary["result"]
    return record


def partition(
    manifest_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    glm_rows: list[dict[str, Any]] | None = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    qwen = latest_successes(qwen_rows)
    glm = latest_successes(glm_rows or [])
    qwen_strict: list[dict[str, Any]] = []
    dual_strict: list[dict[str, Any]] = []
    bands: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    polarities: Counter[str] = Counter()
    qwen_uids: set[str] = set()
    dual_uids: set[str] = set()

    for row in manifest_rows:
        item_id = row["item_id"]
        qwen_row = qwen.get(item_id)
        if qwen_row is None:
            bands["qwen_missing"] += 1
            continue
        qwen_pass = strict_v5_pass(qwen_row["result"])
        glm_row = glm.get(item_id) if glm_rows is not None else None
        glm_pass = (
            strict_v5_pass(glm_row["result"])
            if glm_row is not None
            else False
        )

        if qwen_pass:
            qwen_strict.append(_annotate(row, qwen_row, glm_row))
            qwen_uids.add(str(row["uid"]))

        if glm_rows is None:
            bands["qwen_strict" if qwen_pass else "qwen_reject"] += 1
            continue
        if glm_row is None:
            bands["glm_missing"] += 1
            continue

        band = {
            (True, True): "dual_strict",
            (True, False): "qwen_only",
            (False, True): "glm_only",
            (False, False): "dual_reject",
        }[(qwen_pass, glm_pass)]
        bands[band] += 1
        if band == "dual_strict":
            dual_strict.append(_annotate(row, qwen_row, glm_row))
            categories[str(row.get("category") or "unknown")] += 1
            polarities[str(row.get("polarity") or "unknown")] += 1
            dual_uids.add(str(row["uid"]))

    summary = {
        "kind": "instructional_conditioned_v5_strict_partition",
        "corpus_action": "none_shadow_only",
        "manifest_items": len(manifest_rows),
        "qwen_v5_valid": len(qwen),
        "glm_v5_valid": len(glm) if glm_rows is not None else None,
        "strict_yes_fields": list(STRICT_YES_FIELDS),
        "qwen_strict_items": len(qwen_strict),
        "qwen_strict_unique_uids": len(qwen_uids),
        "qwen_strict_fraction": (
            len(qwen_strict) / len(manifest_rows) if manifest_rows else 0.0
        ),
        "dual_strict_items": len(dual_strict) if glm_rows is not None else None,
        "dual_strict_unique_uids": (
            len(dual_uids) if glm_rows is not None else None
        ),
        "bands": dict(sorted(bands.items())),
        "dual_by_category": dict(sorted(categories.items())),
        "dual_by_polarity": dict(sorted(polarities.items())),
    }
    return qwen_strict, dual_strict, summary


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen queue: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen-v5", type=Path, required=True)
    parser.add_argument("--qwen-strict-out", type=Path, required=True)
    parser.add_argument("--glm-v5", type=Path)
    parser.add_argument("--dual-strict-out", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if bool(args.glm_v5) != bool(args.dual_strict_out):
        parser.error("--glm-v5 and --dual-strict-out must be supplied together")

    qwen_strict, dual_strict, summary = partition(
        load_jsonl(args.manifest),
        load_jsonl(args.qwen_v5),
        load_jsonl(args.glm_v5) if args.glm_v5 else None,
    )
    write_jsonl(args.qwen_strict_out, qwen_strict)
    summary["qwen_strict_sha256"] = sha256_file(args.qwen_strict_out)
    if args.dual_strict_out:
        write_jsonl(args.dual_strict_out, dual_strict)
        summary["dual_strict_sha256"] = sha256_file(args.dual_strict_out)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
