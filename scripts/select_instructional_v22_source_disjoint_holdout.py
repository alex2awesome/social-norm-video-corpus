#!/usr/bin/env python3
"""Freeze a fresh, source-disjoint V22 instructional holdout."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def stratum(row: dict[str, Any]) -> tuple[str, str]:
    return (
        str(row.get("category") or "unknown"),
        str(row.get("polarity") or "unknown"),
    )


def select(
    rows: list[dict[str, Any]],
    count: int,
    seed: str,
    excluded_uids: set[str],
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    seen_items: set[str] = set()
    for row in rows:
        item_id = str(row.get("item_id") or "")
        uid = str(row.get("uid") or "")
        if not item_id or not uid or not row.get("source_clip"):
            continue
        if item_id in seen_items:
            raise ValueError(f"duplicate item_id: {item_id}")
        seen_items.add(item_id)
        if uid not in excluded_uids:
            pools[stratum(row)].append(row)
    for key, values in pools.items():
        values.sort(key=lambda row: stable_key(f"{seed}:{key}", str(row["item_id"])))
    ordered_strata = sorted(
        pools, key=lambda key: stable_key(f"{seed}:strata", repr(key))
    )
    offsets = {key: 0 for key in ordered_strata}
    used_uids: set[str] = set()
    chosen: list[dict[str, Any]] = []
    while len(chosen) < count:
        progress = False
        for key in ordered_strata:
            values = pools[key]
            while offsets[key] < len(values):
                row = values[offsets[key]]
                offsets[key] += 1
                uid = str(row["uid"])
                if uid in used_uids:
                    continue
                chosen.append(row)
                used_uids.add(uid)
                progress = True
                break
            if len(chosen) == count:
                break
        if not progress:
            raise ValueError("insufficient source-disjoint eligible rows")
    return sorted(
        chosen, key=lambda row: stable_key(f"{seed}:blind", str(row["item_id"]))
    )


def blind_record(row: dict[str, Any], index: int) -> dict[str, Any]:
    output = {
        "audit_index": index,
        "candidate_id": f"v22holdout-{index:04d}",
        "item_id": row["item_id"],
        "uid": row["uid"],
        "pillar": "instructional",
        "source_clip": row["source_clip"],
    }
    source_hash = row.get("source_sha256") or row.get("source_clip_sha256")
    if source_hash:
        output["source_sha256"] = source_hash
    return output


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def promotion_gate(
    count: int,
    audit_tier: str,
    gate_profile: str = "v23_consensus",
) -> dict[str, Any]:
    if audit_tier == "intensive" and count < 100:
        raise ValueError("intensive audit requires at least 100 items")
    if gate_profile == "title_v23_union":
        return {
            "scope": "review_ranking_or_candidate_generation_only",
            "required_manual_coverage": count,
            "title_or_v23_min_precision": 0.75,
            "title_or_v23_min_recall": 0.65,
            "uniform_source_disjoint_sample": True,
            "acceptance_or_rejection_authority": False,
        }
    if gate_profile == "blind_episode_relaxed":
        return {
            "scope": "visual_demo_candidate_generation_and_review_ranking_only",
            "required_manual_coverage": count,
            "qwen_blind_episode_without_completeness_min_precision": 0.80,
            "qwen_blind_episode_without_completeness_min_recall": 0.40,
            "qwen_blind_episode_without_completeness_min_selected": 15,
            "exact_label_symbolic_result_is_exploratory": True,
            "uniform_source_disjoint_sample": True,
            "acceptance_or_rejection_authority": False,
        }
    if gate_profile == "demo_consensus_v2":
        return {
            "scope": "visual_demo_candidate_generation_and_review_ranking_only",
            "required_manual_coverage": count,
            "qwen_v1_v2_consensus_min_precision": 0.80,
            "qwen_v1_v2_consensus_min_recall": 0.40,
            "qwen_v1_v2_consensus_min_selected": 15,
            "post_scale_transfer_audit_required": True,
            "uniform_source_disjoint_sample": True,
            "acceptance_or_rejection_authority": False,
        }
    if audit_tier == "intensive":
        return {
            "scope": "review_ranking_or_candidate_generation_only",
            "required_manual_coverage": count,
            "dual_model_pipeline_min_precision": 0.85,
            "dual_model_pipeline_min_recall": 0.50,
            "dual_model_pipeline_min_selected": 20,
            "source_cluster_exact_min_precision": 0.85,
            "large_format_slice_min_selected": 10,
            "large_format_slice_min_precision": 0.70,
            "acceptance_or_rejection_authority": False,
        }
    return {
        "scope": "review_ranking_or_candidate_generation_only",
        "required_manual_coverage": count,
        "dual_model_pipeline_min_precision": 0.80,
        "dual_model_pipeline_min_recall": 0.50,
        "dual_model_pipeline_min_selected": 8,
        "acceptance_or_rejection_authority": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path, action="append", required=True)
    parser.add_argument(
        "--exclude-jsonl",
        type=Path,
        action="append",
        default=[],
        help="Additional audited selections whose uid fields must be excluded.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--count", type=int, default=36)
    parser.add_argument("--seed", default="instructional-v22-fresh-holdout-v1")
    parser.add_argument(
        "--audit-tier", choices=("initial", "intensive"), default="initial"
    )
    parser.add_argument(
        "--gate-profile",
        choices=(
            "v23_consensus", "title_v23_union", "blind_episode_relaxed",
            "demo_consensus_v2",
        ),
        default="v23_consensus",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite: {args.out}")
    source_rows = read_jsonl(args.source)
    excluded = {
        line.strip()
        for path in args.exclude_uids
        for line in path.read_text().splitlines()
        if line.strip()
    }
    excluded.update(
        str(row["uid"])
        for path in args.exclude_jsonl
        for row in read_jsonl(path)
        if row.get("uid")
    )
    selected = select(source_rows, args.count, args.seed, excluded)
    args.out.mkdir(parents=True)
    semantic = []
    blind = []
    for index, row in enumerate(selected):
        identity = blind_record(row, index)
        blind.append(identity)
        semantic.append({**row, "audit_index": index, "candidate_id": identity["candidate_id"]})
    semantic_path = args.out / "sealed_selection.jsonl"
    blind_path = args.out / "blind_source_manifest.jsonl"
    semantic_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in semantic)
    )
    blind_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind)
    )
    summary = {
        "kind": "instructional_v22_source_disjoint_holdout",
        "status": "preregistered_blind_manual_audit_pending",
        "policy": "shadow_only_no_keep_reject_or_corpus_mutation",
        "seed": args.seed,
        "source_rows": len(source_rows),
        "excluded_uids": len(excluded),
        "selected": len(selected),
        "selected_uids": len({str(row["uid"]) for row in selected}),
        "strata": {
            f"{category}|{polarity}": count
            for (category, polarity), count in sorted(Counter(map(stratum, selected)).items())
        },
        "semantic_fields_in_blind_manifest": False,
        "audit_tier": args.audit_tier,
        "gate_profile": args.gate_profile,
        "promotion_gate": promotion_gate(
            args.count, args.audit_tier, args.gate_profile
        ),
        "artifact_sha256": {
            "source": sha256(args.source),
            "excluded_uid_files": {
                str(path): sha256(path) for path in args.exclude_uids
            },
            "excluded_jsonl_files": {
                str(path): sha256(path) for path in args.exclude_jsonl
            },
            "sealed_selection": sha256(semantic_path),
            "blind_source_manifest": sha256(blind_path),
        },
    }
    (args.out / "preregistration.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
