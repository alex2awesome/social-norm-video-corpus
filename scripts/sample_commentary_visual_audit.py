#!/usr/bin/env python3
"""Select a deterministic, platform-stratified commentary visual-audit cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".mov", ".m4v")
MICROCLIP_SUFFIX = re.compile(r"__\d+__w\d+$")


def stable_key(seed: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def parse_quotas(values: list[str]) -> dict[str, int]:
    quotas: dict[str, int] = {}
    for value in values:
        platform, count = value.split("=", 1)
        quotas[platform] = int(count)
    if not quotas or any(value < 0 for value in quotas.values()):
        raise ValueError("platform quotas must be nonnegative and nonempty")
    return quotas


def normalize_source_uid(value: str) -> str:
    """Collapse item/window identifiers to the retained source-video UID."""
    value = value.strip()
    if value.startswith(("commentary:", "instructional:", "witnessed:")):
        parts = value.split(":")
        if len(parts) >= 2:
            value = parts[1]
    value = value.split(":window:", 1)[0]
    return MICROCLIP_SUFFIX.sub("", value)


def load_exclusions(path: Path | None) -> set[str]:
    if path is None:
        return set()
    return {
        normalize_source_uid(line)
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }


def load_manifest_exclusions(path: Path) -> set[str]:
    """Load source UIDs from either a JSON object/list or JSONL manifest."""
    text = path.read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        values = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        if isinstance(value, dict):
            values = value.get("items") or [value]
        elif isinstance(value, list):
            values = value
        else:
            raise ValueError(f"unsupported exclusion manifest: {path}")
    result = set()
    for row in values:
        if not isinstance(row, dict):
            continue
        value = row.get("uid") or row.get("video_id") or row.get("item_id")
        if value:
            result.add(normalize_source_uid(str(value)))
    return result


def has_video(video_dir: Path, uid: str) -> bool:
    return any((video_dir / f"{uid}{suffix}").is_file() for suffix in VIDEO_SUFFIXES)


def candidate_from_record(record: dict, statement_index: int) -> dict:
    statement = record["statements"][statement_index]
    provenance = record.get("provenance") or {}
    return {
        "uid": record["video_id"],
        "item_id": f"commentary:{record['video_id']}:{statement_index}",
        "item_index": statement_index,
        "title": record.get("title"),
        "category": record.get("category"),
        "source_platform": record.get("source")
        or provenance.get("platform")
        or record["video_id"].split("__", 1)[0],
        "query_source": provenance.get("query_source"),
        "found_by_query": provenance.get("found_by_query"),
        "signal": statement.get("signal"),
        "decision": "unreviewed",
        "normalized_behavior": statement.get("quote"),
        "normalized_norm": statement.get("norm"),
        "behavior_evidence_quote": statement.get("quote"),
        "stance_evidence_quote": statement.get("quote"),
        "detector_start_sec": statement.get("start"),
        "detector_end_sec": statement.get("end"),
        "url": record.get("url"),
    }


def load_candidates(
    discussion_dir: Path,
    video_dir: Path,
    exclusions: set[str],
    seed: str,
) -> list[dict]:
    candidates = []
    for path in sorted(discussion_dir.glob("*.json")):
        record = json.loads(path.read_text())
        uid = record.get("video_id")
        if not uid or uid in exclusions or not has_video(video_dir, uid):
            continue
        valid = [
            index
            for index, statement in enumerate(record.get("statements") or [])
            if str(statement.get("quote") or "").strip()
            and isinstance(statement.get("start"), (int, float))
            and isinstance(statement.get("end"), (int, float))
            and statement["end"] >= statement["start"]
        ]
        if not valid:
            continue
        statement_index = min(
            valid,
            key=lambda index: stable_key(seed, f"{uid}:{index}"),
        )
        candidates.append(candidate_from_record(record, statement_index))
    return candidates


def diversity_sample(rows: list[dict], count: int, seed: str) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in rows:
        group = (
            str(row.get("query_source") or "unknown"),
            str(row.get("category") or "unknown"),
            str(row.get("signal") or "unknown"),
        )
        groups[group].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: stable_key(seed, row["item_id"]))

    ordered_groups = sorted(
        groups,
        key=lambda group: stable_key(seed, "\0".join(group)),
    )
    selected: list[dict] = []
    round_index = 0
    while len(selected) < count:
        added = False
        for group in ordered_groups:
            rows_in_group = groups[group]
            if round_index < len(rows_in_group):
                selected.append(rows_in_group[round_index])
                added = True
                if len(selected) == count:
                    break
        if not added:
            break
        round_index += 1
    return selected


def select(
    candidates: list[dict],
    quotas: dict[str, int],
    seed: str,
) -> tuple[list[dict], dict]:
    by_platform: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        by_platform[row["source_platform"]].append(row)
    selected = []
    availability = {}
    for platform, quota in quotas.items():
        pool = by_platform.get(platform, [])
        availability[platform] = len(pool)
        picked = diversity_sample(pool, quota, f"{seed}:{platform}")
        if len(picked) != quota:
            raise ValueError(
                f"platform {platform} has {len(pool)} candidates, needs {quota}"
            )
        selected.extend(picked)
    selected.sort(
        key=lambda row: (
            list(quotas).index(row["source_platform"]),
            stable_key(seed, row["item_id"]),
        )
    )
    return selected, availability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discussion-dir", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path)
    parser.add_argument(
        "--exclude-manifest",
        type=Path,
        action="append",
        default=[],
        help="Additional JSON/JSONL manifests whose source UIDs must be excluded.",
    )
    parser.add_argument("--platform-quota", action="append", required=True)
    parser.add_argument("--seed", default="commentary-visual-audit-v3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    quotas = parse_quotas(args.platform_quota)
    exclusions = load_exclusions(args.exclude_uids)
    for path in args.exclude_manifest:
        exclusions.update(load_manifest_exclusions(path))
    candidates = load_candidates(
        args.discussion_dir, args.video_dir, exclusions, args.seed
    )
    selected, availability = select(candidates, quotas, args.seed)
    manifest = {
        "audit": "commentary_visual_source_disjoint_v3",
        "selection": (
            "deterministic platform quotas with round-robin diversity over "
            "query_source x category x signal"
        ),
        "seed": args.seed,
        "excluded_uids": len(exclusions),
        "available_by_platform": availability,
        "quotas": quotas,
        "items": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "items": len(selected),
                "by_platform": {
                    platform: sum(
                        row["source_platform"] == platform for row in selected
                    )
                    for platform in quotas
                },
                "available_by_platform": availability,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
