#!/usr/bin/env python3
"""Build labeling-function matrices and diagnostics (roadmap section 12.7).

Reads standardized LF records (labeling_functions_v1), groups them into one
matrix per (pillar, target), and reports per-LF coverage and polarity,
pairwise overlap/conflict/correlation, correlated same-family pairs, and —
when a manual gold ledger is supplied — per-LF precision/recall with Wilson
lower bounds.  Diagnostics only: nothing here labels or mutates the corpus.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from weaksup.labeling_functions_v1 import validate_lf_record
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from weaksup.labeling_functions_v1 import validate_lf_record


CORRELATED_FAMILY_FLAG_THRESHOLD = 0.5


def wilson_lower(successes: int, total: int, z: float = 1.96) -> float | None:
    if total <= 0:
        return None
    p = successes / total
    denom = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return max(0.0, (center - margin) / denom)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def build_matrices(
    records: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Group records into {(pillar, target): {items, lfs, votes}}.

    ``votes`` maps (item_id, lf_id) -> vote.  A duplicate (item, lf) vote with
    a different value is an input error, not something to silently overwrite.
    """
    matrices: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        validate_lf_record(record)
        key = (record["pillar"], record["target"])
        matrix = matrices.setdefault(
            key, {"items": set(), "lfs": {}, "votes": {}}
        )
        cell = (record["item_id"], record["lf_id"])
        if cell in matrix["votes"] and matrix["votes"][cell] != record["vote"]:
            raise ValueError(f"conflicting duplicate vote for {cell}")
        matrix["items"].add(record["item_id"])
        matrix["lfs"][record["lf_id"]] = record["family"]
        matrix["votes"][cell] = record["vote"]
    return matrices


def _pair_stats(matrix: dict[str, Any], lf_a: str, lf_b: str) -> dict[str, Any]:
    both = agree = conflict = 0
    xs, ys = [], []
    for item in matrix["items"]:
        va = matrix["votes"].get((item, lf_a), 0)
        vb = matrix["votes"].get((item, lf_b), 0)
        xs.append(va)
        ys.append(vb)
        if va != 0 and vb != 0:
            both += 1
            if va == vb:
                agree += 1
            else:
                conflict += 1
    n = len(xs)
    correlation = None
    if n > 1:
        mean_x, mean_y = sum(xs) / n, sum(ys) / n
        var_x = sum((x - mean_x) ** 2 for x in xs)
        var_y = sum((y - mean_y) ** 2 for y in ys)
        if var_x > 0 and var_y > 0:
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            correlation = cov / math.sqrt(var_x * var_y)
    return {
        "lf_a": lf_a,
        "lf_b": lf_b,
        "overlap": both / n if n else 0.0,
        "agree": agree,
        "conflict": conflict,
        "conflict_rate": conflict / both if both else None,
        "correlation": correlation,
    }


def matrix_diagnostics(
    matrix: dict[str, Any], gold: dict[str, int] | None = None
) -> dict[str, Any]:
    """Coverage, polarity, pairwise, correlated-family, and gold diagnostics.

    ``gold`` maps item_id -> +1/-1 manual audit labels for this target.
    """
    items = sorted(matrix["items"])
    lfs = sorted(matrix["lfs"])
    n = len(items)
    per_lf = []
    for lf_id in lfs:
        votes = [matrix["votes"].get((item, lf_id), 0) for item in items]
        positive = sum(v == 1 for v in votes)
        negative = sum(v == -1 for v in votes)
        row: dict[str, Any] = {
            "lf_id": lf_id,
            "family": matrix["lfs"][lf_id],
            "coverage": (positive + negative) / n if n else 0.0,
            "positive": positive,
            "negative": negative,
            "abstain": n - positive - negative,
        }
        if gold:
            tp = fp = fn = 0
            for item in items:
                label = gold.get(item)
                if label is None:
                    continue
                vote = matrix["votes"].get((item, lf_id), 0)
                if vote == 1 and label == 1:
                    tp += 1
                elif vote == 1 and label == -1:
                    fp += 1
                elif vote != 1 and label == 1:
                    fn += 1
            selected = tp + fp
            row.update(
                gold_tp=tp,
                gold_fp=fp,
                gold_fn=fn,
                gold_precision=tp / selected if selected else None,
                gold_recall=tp / (tp + fn) if tp + fn else None,
                gold_precision_wilson_lower=wilson_lower(tp, selected),
            )
        per_lf.append(row)

    pairs = [
        _pair_stats(matrix, lfs[i], lfs[j])
        for i in range(len(lfs))
        for j in range(i + 1, len(lfs))
    ]
    correlated_same_family = [
        pair
        for pair in pairs
        if pair["correlation"] is not None
        and pair["correlation"] >= CORRELATED_FAMILY_FLAG_THRESHOLD
        and matrix["lfs"][pair["lf_a"]] == matrix["lfs"][pair["lf_b"]]
    ]
    return {
        "items": n,
        "lfs": len(lfs),
        "per_lf": per_lf,
        "pairwise": pairs,
        "correlated_same_family_pairs": correlated_same_family,
        "gold_items": len(gold) if gold else 0,
        "policy": "diagnostics_only_no_labels_no_mutation",
    }


def load_gold(path: Path, pillar: str, target: str) -> dict[str, int]:
    gold: dict[str, int] = {}
    for row in iter_jsonl(path):
        if row.get("pillar") != pillar or row.get("target") != target:
            continue
        label = row.get("label")
        if label not in (-1, 1):
            raise ValueError(f"gold label must be +1/-1, got {label!r}")
        gold[row["item_id"]] = label
    return gold


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--gold", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    matrices = build_matrices(iter_jsonl(args.records))
    report = {}
    for (pillar, target), matrix in sorted(matrices.items()):
        gold = load_gold(args.gold, pillar, target) if args.gold else None
        report[f"{pillar}:{target}"] = matrix_diagnostics(matrix, gold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: {"items": value["items"], "lfs": value["lfs"]} for key, value in report.items()},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
