#!/usr/bin/env python3
"""Select the fresh source-disjoint Instructional V4 transfer cohort."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


MANUAL_FIELDS = (
    "audit_index", "candidate_id", "item_id", "uid", "visual_demo",
    "temporal_state_change", "recipient_response_or_coordinated_trajectory",
    "segment_start_frame", "segment_end_frame", "label_alignment",
    "manual_description", "manual_note",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def one_per_source(
    rows: list[dict[str, Any]], excluded: set[str], seed: str
) -> list[dict[str, Any]]:
    by_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    item_ids: set[str] = set()
    for row in rows:
        uid = str(row.get("uid") or "")
        item_id = str(row.get("item_id") or "")
        if not uid or not item_id or not row.get("source_clip"):
            continue
        if item_id in item_ids:
            raise ValueError("population contains duplicate item_id")
        item_ids.add(item_id)
        if uid not in excluded:
            by_uid[uid].append(row)
    output = []
    for uid, values in by_uid.items():
        output.append(min(values, key=lambda row: stable(seed, str(row["item_id"]))))
    return output


def stratum(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("source_platform") or str(row["uid"]).split("__", 1)[0]),
        str(row.get("category") or "unknown"),
        str(row.get("polarity") or "unknown"),
    )


def select(
    rows: list[dict[str, Any]], excluded: set[str], count: int, seed: str
) -> list[dict[str, Any]]:
    unique = one_per_source(rows, excluded, seed)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in unique:
        groups[stratum(row)].append(row)
    for key, values in groups.items():
        values.sort(key=lambda row: stable(f"{seed}:{key}", str(row["item_id"])))
    keys = sorted(groups, key=lambda key: stable(f"{seed}:strata", repr(key)))
    offsets = {key: 0 for key in keys}
    selected = []
    while len(selected) < count:
        progressed = False
        for key in keys:
            if offsets[key] < len(groups[key]):
                selected.append(groups[key][offsets[key]])
                offsets[key] += 1
                progressed = True
            if len(selected) == count:
                break
        if not progressed:
            raise ValueError("insufficient fresh instructional sources")
    return sorted(selected, key=lambda row: stable(f"{seed}:blind", str(row["item_id"])))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", default="instructional-temporal-critic-v4-fresh-transfer")
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    excluded = {line.strip() for line in args.exclude_uids.read_text().splitlines() if line.strip()}
    selected = select(read_jsonl(args.population), excluded, args.count, args.seed)
    args.out_dir.mkdir(parents=True)
    semantic = []
    blind = []
    for index, row in enumerate(selected):
        candidate_id = f"instructional-v4-fresh-{index:04d}"
        semantic.append({**row, "audit_index": index, "candidate_id": candidate_id})
        blind.append({
            "audit_index": index, "candidate_id": candidate_id,
            "item_id": row["item_id"], "uid": row["uid"],
            "pillar": "instructional", "source_clip": row["source_clip"],
            "source_clip_sha256": row.get("source_clip_sha256") or row.get("source_sha256"),
        })
    write_jsonl(args.out_dir / "sealed_selection.jsonl", semantic)
    write_jsonl(args.out_dir / "blind_source_manifest.jsonl", blind)
    with (args.out_dir / "manual_gold.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANUAL_FIELDS, delimiter="\t")
        writer.writeheader()
        for row in semantic:
            writer.writerow({key: row.get(key, "") for key in MANUAL_FIELDS})
    summary = {
        "kind": "instructional_temporal_critic_v4_fresh_selection",
        "seed": args.seed,
        "selected": len(selected),
        "source_disjoint": len({row["uid"] for row in selected}) == len(selected),
        "excluded_prior_uids": len(excluded),
        "strata": {"|".join(key): value for key, value in sorted(Counter(stratum(row) for row in selected).items())},
        "human_gold_must_be_sealed_before_model_scoring": True,
        "manual_review_every_model_output": True,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }
    (args.out_dir / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
