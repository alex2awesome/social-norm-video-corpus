#!/usr/bin/env python3
"""Freeze source-disjoint cohorts for conditioned-v5 instructional audit.

The `controls` phase must run before GLM scores are inspected. The `final`
phase later selects the primary dual-positive and disagreement cohorts and
preserves the already-frozen Qwen-reject control cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

if __package__:
    from scripts.select_instructional_v5_strict_candidates import (
        latest_successes,
        strict_v5_pass,
    )
else:
    from select_instructional_v5_strict_candidates import (
        latest_successes,
        strict_v5_pass,
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def balanced_source_disjoint(
    rows: list[dict[str, Any]],
    count: int,
    seed: int,
    used_uids: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Round-robin category/polarity strata with deterministic UID sampling."""
    used = used_uids if used_uids is not None else set()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row["uid"]) in used:
            continue
        key = (
            str(row.get("category") or "unknown"),
            str(row.get("polarity") or "unknown"),
        )
        groups[key].append(row)
    for values in groups.values():
        values.sort(key=lambda row: stable_key(seed, str(row["item_id"])))
    keys = sorted(groups, key=lambda key: stable_key(seed, repr(key)))

    selected: list[dict[str, Any]] = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            values = groups[key]
            while values and str(values[0]["uid"]) in used:
                values.pop(0)
            if not values:
                continue
            row = values.pop(0)
            used.add(str(row["uid"]))
            selected.append(row)
            progressed = True
            if len(selected) == count:
                break
        if not progressed:
            break
    return selected


def annotate(
    row: dict[str, Any],
    band: str,
    qwen: dict[str, dict[str, Any]],
    glm: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    record = dict(row)
    record["audit_band"] = band
    record["qwen_v5_conditioned_result"] = qwen[row["item_id"]]["result"]
    if glm is not None and row["item_id"] in glm:
        record["glm_v5_conditioned_result"] = glm[row["item_id"]]["result"]
    return record


def select_controls(
    manifest: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    excluded_uids: set[str],
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    qwen = latest_successes(qwen_rows)
    candidates = [
        row
        for row in manifest
        if row["item_id"] in qwen
        and not strict_v5_pass(qwen[row["item_id"]]["result"])
        and str(row["uid"]) not in excluded_uids
    ]
    return [
        annotate(row, "qwen_reject_control", qwen)
        for row in balanced_source_disjoint(candidates, count, seed)
    ]


def select_final(
    manifest: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    glm_rows: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    excluded_uids: set[str],
    dual_count: int,
    disagreement_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    qwen = latest_successes(qwen_rows)
    glm = latest_successes(glm_rows)
    used = set(excluded_uids)
    used.update(str(row["uid"]) for row in controls)

    def eligible(predicate: Callable[[bool, bool], bool]) -> list[dict[str, Any]]:
        result = []
        for row in manifest:
            item_id = row["item_id"]
            if item_id not in qwen or item_id not in glm:
                continue
            qp = strict_v5_pass(qwen[item_id]["result"])
            gp = strict_v5_pass(glm[item_id]["result"])
            if predicate(qp, gp) and str(row["uid"]) not in used:
                result.append(row)
        return result

    dual = balanced_source_disjoint(
        eligible(lambda qp, gp: qp and gp),
        dual_count,
        seed,
        used,
    )
    disagreements = balanced_source_disjoint(
        eligible(lambda qp, gp: qp and not gp),
        disagreement_count,
        seed + 1,
        used,
    )
    selected = [
        annotate(row, "dual_strict", qwen, glm) for row in dual
    ]
    selected.extend(
        annotate(row, "qwen_strict_glm_reject", qwen, glm)
        for row in disagreements
    )
    selected.extend(
        annotate(row, "qwen_reject_control", qwen, glm)
        for row in controls
    )
    for index, row in enumerate(selected):
        row["audit_index"] = index
    return selected


def write_frozen(path: Path, rows: list[dict[str, Any]]) -> str:
    if path.exists():
        raise SystemExit(f"refusing to overwrite frozen cohort: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("controls", "final"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen-v5", type=Path, required=True)
    parser.add_argument(
        "--glm-v5",
        type=Path,
        action="append",
        help="GLM append-only ledger; repeat when control and main runs are separate.",
    )
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--excluded-uids", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--control-count", type=int, default=30)
    parser.add_argument("--dual-count", type=int, default=60)
    parser.add_argument("--disagreement-count", type=int, default=30)
    args = parser.parse_args()

    manifest = load_jsonl(args.manifest)
    qwen_rows = load_jsonl(args.qwen_v5)
    excluded = set(args.excluded_uids.read_text().split())
    if args.phase == "controls":
        rows = select_controls(
            manifest,
            qwen_rows,
            excluded,
            args.control_count,
            args.seed,
        )
    else:
        if not args.glm_v5 or not args.controls:
            parser.error("final phase requires --glm-v5 and --controls")
        glm_rows = [
            row
            for path in args.glm_v5
            for row in load_jsonl(path)
        ]
        rows = select_final(
            manifest,
            qwen_rows,
            glm_rows,
            load_jsonl(args.controls),
            excluded,
            args.dual_count,
            args.disagreement_count,
            args.seed,
        )
    digest = write_frozen(args.out, rows)
    band_counts: dict[str, int] = {}
    for row in rows:
        band = row["audit_band"]
        band_counts[band] = band_counts.get(band, 0) + 1
    summary = {
        "phase": args.phase,
        "seed": args.seed,
        "items": len(rows),
        "unique_uids": len({str(row["uid"]) for row in rows}),
        "excluded_uids": len(excluded),
        "by_band": dict(sorted(band_counts.items())),
        "sha256": digest,
        "corpus_action": "none_shadow_only",
    }
    if args.summary.exists():
        raise SystemExit(f"refusing to overwrite frozen summary: {args.summary}")
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
