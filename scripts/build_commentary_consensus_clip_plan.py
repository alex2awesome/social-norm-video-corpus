#!/usr/bin/env python3
"""Build bounded, unapproved clip proposals from dual-VLM title retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def proposed_bounds(
    qwen: dict,
    gemma: dict,
    duration_sec: float,
    min_duration: float = 6.0,
    max_duration: float = 12.0,
) -> tuple[float, float, str]:
    candidates = []
    for name, row in (("qwen", qwen), ("gemma", gemma)):
        result = row.get("result") or {}
        start = result.get("candidate_start_sec")
        end = result.get("candidate_end_sec")
        if start is None or end is None:
            continue
        start, end = float(start), float(end)
        if 0 <= start < end <= duration_sec + 1:
            candidates.append((end - start, start, end, name))
    if not candidates:
        raise ValueError("dual consensus row has no valid candidate bounds")
    _, start, end, proposer = min(candidates)
    center = (start + end) / 2
    target = max(min_duration, min(max_duration, (end - start) + 4.0))
    start = max(0.0, center - target / 2)
    end = min(duration_sec, start + target)
    start = max(0.0, end - target)
    return round(start, 3), round(end, 3), proposer


def build(
    semantic_path: Path,
    qwen_path: Path,
    gemma_path: Path,
    evaluation_path: Path,
) -> list[dict]:
    semantic = {
        int(row["audit_index"]): row for row in read_jsonl(semantic_path)
    }
    qwen = {int(row["audit_index"]): row for row in read_jsonl(qwen_path)}
    gemma = {int(row["audit_index"]): row for row in read_jsonl(gemma_path)}
    evaluation = json.loads(evaluation_path.read_text())
    metric = evaluation["slices"]["clear_94_validation"]["rules"][
        "dual_retrieval_exact"
    ]["usable_any_route"]
    selected = list(map(int, metric["selected_indices"]))
    output = []
    for index in selected:
        source = semantic[index]
        duration = float(source["media"]["duration_sec"])
        start, end, proposer = proposed_bounds(
            qwen[index], gemma[index], duration
        )
        qresult = qwen[index].get("result") or {}
        gresult = gemma[index].get("result") or {}
        output.append(
            {
                "audit_index": index,
                "candidate_id": source["candidate_id"],
                "item_id": source["item_id"],
                "uid": source["uid"],
                "title": source["norm"],
                "source_path": source["source_path_resolved"],
                "source_duration_sec": duration,
                "proposed_start_sec": start,
                "proposed_end_sec": end,
                "proposal_source": f"shorter_{proposer}_interval_plus_context",
                "qwen_bounds_sec": [
                    qresult.get("candidate_start_sec"),
                    qresult.get("candidate_end_sec"),
                ],
                "gemma_bounds_sec": [
                    gresult.get("candidate_start_sec"),
                    gresult.get("candidate_end_sec"),
                ],
                "selection_rule": "dual_retrieval_exact_clear_validation",
                "approval_status": "unreviewed_candidate",
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = build(args.semantic, args.qwen, args.gemma, args.evaluation)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in output)
    )


if __name__ == "__main__":
    main()
