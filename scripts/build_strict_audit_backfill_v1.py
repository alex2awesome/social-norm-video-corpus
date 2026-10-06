#!/usr/bin/env python3
"""Inventory every corpus video and measure strict-audit coverage.

The output is an append-only shadow ledger.  It does not change corpus media,
metadata, routing, or labels.  A media row is keyed by its relative path; label
rows retain the pillar's existing item ids.  This separation is important for
commentary (many statements can refer to one retained source video) and for
witnessed clips (more than one reaction record can refer to one cut clip).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


VIDEO_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".m4v"}
POSITIVE_PILLARS = ("instructional", "witnessed", "commentary")
WITNESSED_CLIP = re.compile(r"^clip_(\d+)\.[^.]+$")


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def media_id(relative_path: str) -> str:
    return "media:" + hashlib.sha256(relative_path.encode()).hexdigest()[:20]


def media_row(path: Path, root: Path, pillar: str, uid: str) -> dict[str, Any]:
    rel = relative(path, root)
    stat = path.stat()
    return {
        "media_id": media_id(rel),
        "pillar": pillar,
        "uid": uid,
        "media_path": rel,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "label_item_ids": [],
        "label_count": 0,
        "metadata_present": False,
        "low_level_visual_complete": False,
        "low_level_visual_windows": 0,
        "manual_strict_complete": False,
        "manual_judgment_count": 0,
        "manual_decisions": {},
        "pillar_shadow_complete": False,
        "pillar_shadow_rules": [],
        "automatic_acceptance": False,
        "automatic_rejection": False,
        "corpus_disposition": None,
        "delete_media": False,
    }


def scan_instructional(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    media: dict[str, dict[str, Any]] = {}
    labels: list[dict[str, Any]] = []
    base = root / "data" / "instructional"
    for source_dir in sorted(path for path in base.glob("*") if path.is_dir()):
        metadata_path = source_dir / "metadata.json"
        metadata = read_json(metadata_path)
        uid = str((metadata or {}).get("video_id") or source_dir.name)
        for path in sorted(
            item for item in source_dir.iterdir()
            if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
        ):
            row = media_row(path, root, "instructional", uid)
            row["metadata_present"] = metadata is not None
            media[row["media_path"]] = row
        if metadata is None:
            continue
        provenance = metadata.get("provenance") or {}
        for index, raw in enumerate(metadata.get("demos") or []):
            demo = raw if isinstance(raw, dict) else {}
            clip_name = demo.get("clip")
            path = source_dir / str(clip_name) if clip_name else None
            rel = relative(path, root) if path and path.is_file() else None
            item_id = f"instructional:{uid}:{index}"
            label = {
                "item_id": item_id,
                "pillar": "instructional",
                "uid": uid,
                "item_index": index,
                "media_path": rel,
                "media_present": bool(rel),
                "metadata_path": relative(metadata_path, root),
                "category": metadata.get("category") or provenance.get("category"),
                "query_source": provenance.get("query_source"),
                "found_by_query": provenance.get("found_by_query"),
                "polarity": demo.get("polarity"),
                "norm": demo.get("norm"),
                "start_sec": demo.get("start"),
                "end_sec": demo.get("end"),
                "metadata_contract_complete": bool(
                    rel and demo.get("norm") and demo.get("polarity")
                ),
            }
            labels.append(label)
            if rel in media:
                media[rel]["label_item_ids"].append(item_id)
    return list(media.values()), labels


def scan_witnessed(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    media: dict[str, dict[str, Any]] = {}
    labels: list[dict[str, Any]] = []
    base = root / "data" / "hits"
    for source_dir in sorted(path for path in base.glob("*") if path.is_dir()):
        metadata_path = source_dir / "metadata.json"
        metadata = read_json(metadata_path)
        uid = str((metadata or {}).get("video_id") or source_dir.name)
        clip_paths: dict[int, Path] = {}
        for path in sorted(
            item for item in source_dir.iterdir()
            if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
        ):
            row = media_row(path, root, "witnessed", uid)
            row["metadata_present"] = metadata is not None
            media[row["media_path"]] = row
            match = WITNESSED_CLIP.match(path.name)
            if match:
                clip_paths[int(match.group(1))] = path
        if metadata is None:
            continue
        provenance = metadata.get("provenance") or {}
        scene = provenance.get("scene") or {}
        for ordinal, raw in enumerate(metadata.get("reactions") or []):
            reaction = raw if isinstance(raw, dict) else {}
            clip_index = reaction.get("clip_idx")
            if not isinstance(clip_index, int):
                clip_index = ordinal
            path = clip_paths.get(clip_index)
            rel = relative(path, root) if path else None
            window = reaction.get("clip_window") or [None, None]
            item_id = f"witnessed:{uid}:{ordinal}"
            label = {
                "item_id": item_id,
                "clip_alias_id": f"witnessed:{uid}:clip_{clip_index}",
                "pillar": "witnessed",
                "uid": uid,
                "item_index": ordinal,
                "clip_index": clip_index,
                "media_path": rel,
                "media_present": bool(rel),
                "metadata_path": relative(metadata_path, root),
                "category": metadata.get("category") or provenance.get("category"),
                "query_source": provenance.get("query_source"),
                "found_by_query": provenance.get("found_by_query"),
                "reaction_tag": reaction.get("tag"),
                "norm": reaction.get("norm"),
                "reaction_text": reaction.get("matched_text") or reaction.get("phrase"),
                "reactor_role": scene.get("reactor_role"),
                "violator_role": scene.get("violator_role"),
                "start_sec": window[0] if len(window) > 0 else None,
                "end_sec": window[1] if len(window) > 1 else None,
                "metadata_contract_complete": bool(
                    rel
                    and (reaction.get("matched_text") or reaction.get("phrase"))
                    and reaction.get("norm")
                ),
            }
            labels.append(label)
            if rel in media:
                media[rel]["label_item_ids"].append(item_id)
    return list(media.values()), labels


def scan_commentary(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    media: dict[str, dict[str, Any]] = {}
    labels: list[dict[str, Any]] = []
    metadata_dir = root / "data" / "discussion"
    media_dir = root / "data" / "discussion_video"
    metadata_by_uid: dict[str, tuple[Path, dict[str, Any]]] = {}
    # Index the flat media directory once. Repeated ``glob(f"{uid}.*")`` calls
    # are effectively quadratic on large network filesystems.
    commentary_paths = sorted(
        item for item in media_dir.glob("*")
        if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
    )
    media_by_uid: dict[str, Path] = {}
    for path in commentary_paths:
        media_by_uid.setdefault(path.stem, path)
    for metadata_path in sorted(metadata_dir.glob("*.json")):
        metadata = read_json(metadata_path)
        if metadata is None:
            continue
        uid = str(metadata.get("video_id") or metadata_path.stem)
        metadata_by_uid[uid] = (metadata_path, metadata)
        path = media_by_uid.get(uid)
        if path:
            rel = relative(path, root)
            row = media_row(path, root, "commentary", uid)
            row["metadata_present"] = True
            media[rel] = row
        provenance = metadata.get("provenance") or {}
        for index, raw in enumerate(metadata.get("statements") or []):
            statement = raw if isinstance(raw, dict) else {}
            item_id = f"commentary:{uid}:{index}"
            rel = relative(path, root) if path else None
            label = {
                "item_id": item_id,
                "pillar": "commentary",
                "uid": uid,
                "item_index": index,
                "media_path": rel,
                "media_present": bool(rel),
                "metadata_path": relative(metadata_path, root),
                "category": metadata.get("category") or provenance.get("category"),
                "query_source": provenance.get("query_source"),
                "found_by_query": provenance.get("found_by_query"),
                "signal": statement.get("signal"),
                "norm": statement.get("norm"),
                "quote": statement.get("quote"),
                "start_sec": statement.get("start"),
                "end_sec": statement.get("end"),
                "metadata_contract_complete": bool(
                    statement.get("quote") and statement.get("norm")
                ),
            }
            labels.append(label)
            if rel in media:
                media[rel]["label_item_ids"].append(item_id)
    for path in commentary_paths:
        uid = path.stem
        rel = relative(path, root)
        if rel not in media:
            row = media_row(path, root, "commentary", uid)
            row["metadata_present"] = uid in metadata_by_uid
            media[rel] = row
    return list(media.values()), labels


def scan_negatives(root: Path) -> list[dict[str, Any]]:
    base = root / "data" / "negatives"
    rows = []
    for path in sorted(
        item for item in base.rglob("*")
        if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES
    ):
        uid = path.parent.name
        row = media_row(path, root, "negative", uid)
        row["metadata_present"] = (path.parent / "metadata.json").is_file()
        rows.append(row)
    return rows


def load_low_level_coverage(root: Path) -> tuple[set[str], Counter[str]]:
    run = root / "data" / "shadow_scores" / "20260724_full_corpus_v1"
    successful: set[str] = set()
    by_path: Counter[str] = Counter()
    for pillar in POSITIVE_PILLARS:
        manifest = {
            str(row.get("item_id")): row
            for row in iter_jsonl(run / f"{pillar}_manifest.jsonl")
            if row.get("item_id")
        }
        for score in iter_jsonl(run / f"{pillar}_low_level_canonical.jsonl"):
            if score.get("error") is not None or not score.get("low_level"):
                continue
            item_id = str(score.get("item_id") or "")
            if not item_id:
                continue
            successful.add(item_id)
            source = manifest.get(item_id, {}).get("source_clip")
            if source:
                by_path[relative(Path(str(source)), root)] += 1
    # Append-only low-level backfill shards use the current item ids and carry
    # their source path directly.  Include every successful shard without
    # rewriting the canonical 20260724 outputs.
    queue_root = (
        root
        / "data"
        / "shadow_scores"
        / "20260811_strict_audit_backfill_queues_v1"
    )
    for path in sorted(queue_root.glob("low_level_shard_*.jsonl")):
        for score in iter_jsonl(path):
            if score.get("error") is not None or not score.get("low_level"):
                continue
            item_id = str(score.get("item_id") or "")
            if not item_id:
                continue
            successful.add(item_id)
            source = score.get("source_clip") or score.get("source_path")
            if source:
                by_path[relative(Path(str(source)), root)] += 1
    return successful, by_path


def load_id_coverage(paths: Iterable[Path], required_field: str | None = None) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for row in iter_jsonl(path):
            if row.get("error") is not None:
                continue
            if required_field and not row.get(required_field):
                continue
            item_id = row.get("item_id")
            if item_id:
                result.add(str(item_id))
    return result


def load_manual_judgments(db_path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    if not db_path.is_file():
        return result
    connection = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        for table in ("judgments", "witnessed_judgments", "commentary_judgments"):
            try:
                rows = connection.execute(f"SELECT item_id,decision FROM {table}")
            except sqlite3.OperationalError:
                continue
            for item_id, decision in rows:
                result[str(item_id)].append(str(decision))
    finally:
        connection.close()
    return result


def load_calibration_judgments(root: Path) -> dict[str, list[str]]:
    """Load compact, frozen manual calibrations without changing audit.db."""
    result: dict[str, list[str]] = defaultdict(list)
    for review_path in sorted(
        (root / "audit_runs").glob(
            "*_strict_audit_calibration_v*/manual_post_reveal_review.jsonl"
        )
    ):
        selection_path = review_path.parent / "sealed_selection.jsonl"
        if not selection_path.is_file():
            continue
        selection = {
            int(row["audit_index"]): str(row["item_id"])
            for row in iter_jsonl(selection_path)
            if row.get("audit_index") is not None and row.get("item_id")
        }
        for row in iter_jsonl(review_path):
            try:
                item_id = selection[int(row["audit_index"])]
            except (KeyError, TypeError, ValueError):
                continue
            decision = row.get("strict_decision")
            if decision:
                result[item_id].append(str(decision))
    return result


def enrich(
    root: Path,
    media: list[dict[str, Any]],
    labels: list[dict[str, Any]],
    audit_db: Path,
) -> None:
    low_ids, low_paths = load_low_level_coverage(root)
    score_root = root / "data" / "shadow_scores"
    instructional = load_id_coverage(
        [score_root / "20260806_instructional_combined_review_tiers_v1" / "demo_scores.jsonl"]
    )
    witnessed_authority = load_id_coverage(
        [score_root / "20260806_witnessed_non_destructive_review_routes_v2" / "authority_scores.jsonl"]
    )
    witnessed_asr = load_id_coverage(
        [score_root / "20260806_witnessed_video_asr_v3" / "qwen_video_asr_v3.jsonl"]
    )
    manual = load_manual_judgments(audit_db)
    for item_id, decisions in load_calibration_judgments(root).items():
        for decision in decisions:
            if decision not in manual[item_id]:
                manual[item_id].append(decision)
    by_id = {str(row["item_id"]): row for row in labels}
    by_path = {str(row["media_path"]): row for row in media}
    for label in labels:
        item_id = str(label["item_id"])
        aliases = {item_id}
        if label.get("clip_alias_id"):
            aliases.add(str(label["clip_alias_id"]))
        label["low_level_visual_complete"] = any(alias in low_ids for alias in aliases)
        label["manual_decisions"] = [
            decision for alias in aliases for decision in manual.get(alias, [])
        ]
        label["manual_strict_complete"] = bool(label["manual_decisions"])
        rules = []
        if item_id in instructional:
            rules.append("instructional_combined_review_tiers_v1")
        if any(alias in witnessed_authority for alias in aliases):
            rules.append("witnessed_authority_reaction_text_cues_v3")
        if any(alias in witnessed_asr for alias in aliases):
            rules.append("witnessed_video_asr_v3")
        label["pillar_shadow_rules"] = rules
        label["pillar_shadow_complete"] = bool(rules)
        label["strict_audit_complete"] = label["manual_strict_complete"]
        label["strict_audit_needed"] = not label["strict_audit_complete"]
        label["automatic_acceptance"] = False
        label["automatic_rejection"] = False
        label["corpus_disposition"] = None
        label["delete_media"] = False
    for row in media:
        path = str(row["media_path"])
        linked = [by_id[item_id] for item_id in row["label_item_ids"] if item_id in by_id]
        row["label_count"] = len(row["label_item_ids"])
        row["low_level_visual_windows"] = low_paths[path]
        row["low_level_visual_complete"] = bool(low_paths[path]) or any(
            label["low_level_visual_complete"] for label in linked
        )
        decisions = [decision for label in linked for decision in label["manual_decisions"]]
        row["manual_judgment_count"] = len(decisions)
        row["manual_decisions"] = dict(sorted(Counter(decisions).items()))
        row["manual_strict_complete"] = bool(linked) and all(
            label["manual_strict_complete"] for label in linked
        )
        rules = sorted({rule for label in linked for rule in label["pillar_shadow_rules"]})
        row["pillar_shadow_rules"] = rules
        row["pillar_shadow_complete"] = bool(linked) and all(
            label["pillar_shadow_complete"] for label in linked
        )
        row["strict_audit_complete"] = row["manual_strict_complete"]
        row["strict_audit_needed"] = bool(linked) and not row["strict_audit_complete"]
        row["unlabeled_media"] = not linked


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    result = Counter(row["pillar"] for row in rows if row.get(field))
    return {key: result.get(key, 0) for key in sorted({row["pillar"] for row in rows})}


def build(root: Path, out: Path, audit_db: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite frozen audit run: {out}")
    instructional_media, instructional_labels = scan_instructional(root)
    witnessed_media, witnessed_labels = scan_witnessed(root)
    commentary_media, commentary_labels = scan_commentary(root)
    media = instructional_media + witnessed_media + commentary_media + scan_negatives(root)
    labels = instructional_labels + witnessed_labels + commentary_labels
    media.sort(key=lambda row: (row["pillar"], row["media_path"]))
    labels.sort(key=lambda row: (row["pillar"], row["item_id"]))
    enrich(root, media, labels, audit_db)
    out.mkdir(parents=True)
    media_hash = write_jsonl(out / "media_manifest.jsonl", media)
    label_hash = write_jsonl(out / "label_manifest.jsonl", labels)
    queue = [row for row in labels if row["media_present"] and row["strict_audit_needed"]]
    queue_hash = write_jsonl(out / "strict_audit_queue.jsonl", queue)
    media_counts = Counter(row["pillar"] for row in media)
    label_counts = Counter(row["pillar"] for row in labels)
    summary = {
        "schema_version": 1,
        "kind": "strict_audit_backfill_coverage_v1",
        "policy": "append_only_shadow_no_accept_reject_or_delete",
        "media_items": len(media),
        "media_by_pillar": dict(sorted(media_counts.items())),
        "weak_label_items": len(labels),
        "weak_labels_by_pillar": dict(sorted(label_counts.items())),
        "media_with_low_level_visual": counts(media, "low_level_visual_complete"),
        "media_with_all_labels_manually_audited": counts(media, "manual_strict_complete"),
        "media_needing_strict_audit": counts(media, "strict_audit_needed"),
        "unlabeled_media": counts(media, "unlabeled_media"),
        "labels_with_low_level_visual": counts(labels, "low_level_visual_complete"),
        "labels_with_pillar_shadow": counts(labels, "pillar_shadow_complete"),
        "labels_with_manual_strict_audit": counts(labels, "manual_strict_complete"),
        "strict_audit_queue_items": len(queue),
        "manifest_sha256": {
            "media": media_hash,
            "labels": label_hash,
            "strict_audit_queue": queue_hash,
        },
        "corpus_mutated": False,
        "automatic_acceptance": False,
        "automatic_rejection": False,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audit-db", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    audit_db = args.audit_db or root / "data" / "visual_audit" / "audit.db"
    summary = build(root, args.out, audit_db)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
