#!/usr/bin/env python3
"""Select a cue-stratified, source-disjoint audit of witnessed staging scores."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def read_exclusions(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        result.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    return result


def read_sealed_exclusions(paths: list[Path]) -> set[str]:
    return {
        str(row["uid"])
        for path in paths
        for row in read_jsonl(path)
        if row.get("uid")
    }


def cue_group(row: dict[str, Any]) -> str:
    names = []
    if row.get("title_staging_cue"):
        names.append("title")
    if row.get("transcript_explicit_reveal_cue"):
        names.append("reveal")
    if row.get("transcript_creator_setup_cue"):
        names.append("creator")
    if row.get("title_creator_initiated_candidate_cue"):
        names.append("creator_initiated_title")
    if row.get("title_wwyd_candidate_cue"):
        names.append("wwyd_title")
    return "+".join(names)


def existing_clips(root: Path, uid: str) -> list[Path]:
    return sorted(
        path
        for path in (root / "data" / "hits" / uid).glob("clip_*.mp4")
        if path.is_file() and path.stat().st_size > 0
    )


def select(
    rows: list[dict[str, Any]],
    root: Path,
    exclusions: set[str],
    per_group: int,
    seed: str,
    candidate_field: str = "explicit_staging_candidate",
) -> list[dict[str, Any]]:
    grouped: dict[str, list[tuple[dict[str, Any], Path]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        uid = str(row.get("uid") or "")
        if (
            not row.get(candidate_field)
            or not uid
            or uid in exclusions
            or uid in seen
            or row.get("error")
        ):
            continue
        clips = existing_clips(root, uid)
        if not clips:
            continue
        grouped[cue_group(row)].append((row, clips[0]))
        seen.add(uid)
    rng = random.Random(seed)
    selected: list[tuple[str, dict[str, Any], Path]] = []
    for group in sorted(grouped):
        candidates = grouped[group]
        rng.shuffle(candidates)
        selected.extend((group, row, clip) for row, clip in candidates[:per_group])
    rng.shuffle(selected)
    return [
        {
            "audit_index": index,
            "item_id": f"witnessed_staging:{row['uid']}",
            "uid": row["uid"],
            "clip": str(clip),
            "cue_group": group,
            "title": row.get("title"),
            "channel": row.get("channel"),
            "query": row.get("query"),
            "query_source": row.get("query_source"),
            "matches": row.get("matches"),
            "policy": "manual_audit_before_any_reroute_preserve_original",
        }
        for index, (group, row, clip) in enumerate(selected)
    ]


def sha256_json(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path, action="append", default=[])
    parser.add_argument("--exclude-sealed", type=Path, action="append", default=[])
    parser.add_argument("--per-group", type=int, default=8)
    parser.add_argument("--seed", default="witnessed-staging-audit-v1")
    parser.add_argument(
        "--candidate-field",
        default="explicit_staging_candidate",
        choices=(
            "explicit_staging_candidate",
            "title_creator_initiated_candidate_cue",
            "title_wwyd_candidate_cue",
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    exclusions = read_exclusions(args.exclude_uids) | read_sealed_exclusions(
        args.exclude_sealed
    )
    rows = select(
        read_jsonl(args.scores),
        args.root,
        exclusions,
        args.per_group,
        args.seed,
        args.candidate_field,
    )
    args.out.mkdir(parents=True)
    sealed = args.out / "sealed_selection.jsonl"
    sealed.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows)
    )
    blind_rows = [
        {
            "audit_index": row["audit_index"],
            "item_id": row["item_id"],
            "uid": row["uid"],
            "clip": row["clip"],
        }
        for row in rows
    ]
    blind = args.out / "blind_selection.jsonl"
    blind.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind_rows)
    )
    prereg = {
        "kind": "witnessed_staging_cue_confirmation",
        "seed": args.seed,
        "per_group": args.per_group,
        "candidate_field": args.candidate_field,
        "items": len(rows),
        "groups": {
            group: sum(row["cue_group"] == group for row in rows)
            for group in sorted({row["cue_group"] for row in rows})
        },
        "excluded_source_uids": len(exclusions),
        "sealed_selection_sha256": sha256_json(rows),
        "blind_policy": "cue_group_title_transcript_hidden_during_visual_review",
        "decision_policy": "no_reroute_without_post_reveal_manual_confirmation",
        "corpus_policy": "preserve_all_original_media_and_labels",
    }
    (args.out / "preregistration.json").write_text(
        json.dumps(prereg, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(prereg, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
