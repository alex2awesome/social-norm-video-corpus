#!/usr/bin/env python3
"""Build a source-disjoint 168-clip multimodal benchmark manifest.

The output points at immutable instructional source clips and carries frozen
manual targets for evaluation. It does not modify source media or metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def read_tsv(path: Path) -> dict[int, dict[str, str]]:
    with path.open() as handle:
        return {
            int(row["audit_index"]): row
            for row in csv.DictReader(handle, delimiter="\t")
        }


def yes(value: Any) -> bool:
    return str(value).strip().lower() in {"y", "yes", "true", "1"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_cohort(
    name: str,
    selection_path: Path,
    visual_path: Path,
    semantic_path: Path,
) -> list[dict[str, Any]]:
    visual = read_tsv(visual_path)
    semantic = read_tsv(semantic_path)
    output = []
    for source in read_jsonl(selection_path):
        index = int(source["audit_index"])
        vrow = visual[index]
        srow = semantic[index]
        visual_label = str(vrow.get("visual_demo") or "").upper()
        if visual_label not in {"Y", "N"}:
            raise ValueError(f"{name} index {index}: unresolved visual label")
        exact_value = (
            srow.get("exact_original")
            or srow.get("exact_social_norm")
        )
        usable_value = (
            srow.get("usable_after_relabel")
            or srow.get("relabel_usable")
        )
        if exact_value is None or usable_value is None:
            raise ValueError(f"{name} index {index}: missing semantic target")
        source_clip = source.get("source_clip")
        source_sha = source.get("source_clip_sha256")
        if not source_clip or not source_sha:
            raise ValueError(f"{name} index {index}: missing source identity")
        output.append(
            {
                "item_id": str(source["item_id"]),
                "uid": str(source["uid"]),
                "pillar": "instructional",
                "source_clip": str(source_clip),
                "source_sha256": str(source_sha),
                "audit_cohort": name,
                "audit_index": index,
                "gold_scene_visible": visual_label == "Y",
                "gold_usable": yes(usable_value),
                "gold_exact_social_norm": yes(exact_value),
                "gold_failure_mode": (
                    srow.get("failure_mechanism")
                    or srow.get("failure_mode")
                    or ""
                ),
            }
        )
    return output


def build(audit_root: Path) -> list[dict[str, Any]]:
    specifications = [
        (
            "v15",
            "20260728_instructional_v15_utterance_anchor_prospective_v1",
            "manual_visual_ledger_blind.tsv",
        ),
        (
            "v17",
            "20260729_instructional_v17_causal_transfer_v1",
            "manual_visual_ledger_final.tsv",
        ),
        (
            "v18",
            "20260729_instructional_v18_animation_holdout_v1",
            "manual_visual_ledger_blind.tsv",
        ),
    ]
    rows = []
    for name, directory, visual_name in specifications:
        root = audit_root / directory
        rows.extend(
            build_cohort(
                name,
                root / "audit_selection_semantic.jsonl",
                root / visual_name,
                root / "manual_semantic_ledger.tsv",
            )
        )
    item_ids = [row["item_id"] for row in rows]
    uids = [row["uid"] for row in rows]
    if len(rows) != 168:
        raise ValueError(f"expected 168 audit rows, got {len(rows)}")
    if len(set(item_ids)) != len(item_ids):
        raise ValueError("benchmark contains duplicate item IDs")
    if len(set(uids)) != len(uids):
        raise ValueError("benchmark contains duplicate source UIDs")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("audit_runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = build(args.audit_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    summary = {
        "kind": "instructional_v19_multimodal_benchmark",
        "items": len(rows),
        "cohort_counts": {
            name: sum(row["audit_cohort"] == name for row in rows)
            for name in ("v15", "v17", "v18")
        },
        "visual_positives": sum(row["gold_scene_visible"] for row in rows),
        "usable_positives": sum(row["gold_usable"] for row in rows),
        "exact_positives": sum(row["gold_exact_social_norm"] for row in rows),
        "manifest_sha256": sha256(args.output),
        "corpus_mutated": False,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
