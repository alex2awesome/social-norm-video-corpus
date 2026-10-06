#!/usr/bin/env python3
"""Maintain the append-only ledger for full-corpus visual audits.

This script never changes corpus metadata or routing state. It inventories
candidate datapoints and stores versioned VLM/human judgments in a separate
SQLite database.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path
from typing import Any


def transform_has_audio_removal(transform: dict[str, Any] | None) -> bool:
    """Return whether any transform in a chained repair removed label audio."""
    if not transform:
        return False
    if transform.get("audio_removed") is True or transform.get("output_audio_removed") is True:
        return True
    prior = transform.get("prior_transform")
    return transform_has_audio_removal(prior if isinstance(prior, dict) else None)


def trim_label_transcript(
    item: dict[str, Any], start: float, end: float
) -> list[dict[str, Any]]:
    """Carry label-transcript provenance through a recut repair chain.

    ``recut_start``/``recut_end`` describe positions in the immediate input
    clip and therefore take precedence after the first recut.  Initial visual
    audit manifests instead use ``clip_start``/``clip_end``.  The explicit
    fallbacks keep older repair manifests auditable without silently dropping
    their label evidence.
    """
    output: list[dict[str, Any]] = []
    source_segments = item.get("aligned_transcript") or item.get("source_label_transcript") or []
    for segment in source_segments:
        try:
            segment_start = float(
                segment.get(
                    "recut_start",
                    segment.get("clip_start", segment.get("source_clip_start")),
                )
            )
            segment_end = float(
                segment.get(
                    "recut_end",
                    segment.get("clip_end", segment.get("source_clip_end")),
                )
            )
        except (TypeError, ValueError):
            continue
        if segment_end < start or segment_start > end:
            continue
        output.append(
            {
                "source_clip_start": round(segment_start, 3),
                "source_clip_end": round(segment_end, 3),
                "recut_start": round(segment_start - start, 3),
                "recut_end": round(segment_end - start, 3),
                "text": segment.get("text"),
            }
        )
    return output


SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS items (
    item_id TEXT PRIMARY KEY,
    pillar TEXT NOT NULL,
    uid TEXT NOT NULL,
    item_index INTEGER NOT NULL,
    metadata_path TEXT NOT NULL,
    clip_path TEXT,
    has_clip INTEGER NOT NULL,
    title TEXT,
    category TEXT,
    genre TEXT,
    source_platform TEXT,
    found_by_query TEXT,
    query_source TEXT,
    polarity TEXT,
    norm TEXT,
    start_quote TEXT,
    end_quote TEXT,
    explanation TEXT,
    start_sec REAL,
    end_sec REAL,
    present INTEGER NOT NULL DEFAULT 1,
    source_mtime REAL NOT NULL,
    discovered_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    UNIQUE(pillar, uid, item_index)
);

CREATE INDEX IF NOT EXISTS idx_items_pending
ON items(pillar, present, has_clip, category, polarity, uid, item_index);

CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    pillar TEXT NOT NULL,
    rubric_version TEXT NOT NULL,
    intended_model TEXT NOT NULL,
    selection_strategy TEXT NOT NULL,
    seed TEXT NOT NULL,
    manifest_path TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS batch_items (
    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
    item_id TEXT NOT NULL REFERENCES items(item_id),
    ordinal INTEGER NOT NULL,
    frame_count INTEGER NOT NULL,
    frame_manifest_sha256 TEXT NOT NULL,
    PRIMARY KEY(batch_id, item_id),
    UNIQUE(batch_id, ordinal)
);

CREATE TABLE IF NOT EXISTS judgments (
    judgment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL REFERENCES items(item_id),
    batch_id TEXT REFERENCES batches(batch_id),
    rubric_version TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    frame_manifest_sha256 TEXT NOT NULL,
    pass_index INTEGER NOT NULL DEFAULT 0,
    is_social_norm TEXT NOT NULL,
    visual_demo_present TEXT NOT NULL,
    medium TEXT NOT NULL,
    clip_composition TEXT,
    norm_supported TEXT NOT NULL,
    polarity_supported TEXT NOT NULL,
    explanation_supports_norm TEXT NOT NULL,
    dialogue_grounded TEXT,
    visual_norm_supported_without_audio TEXT,
    narration_leak TEXT,
    label_leak_visible TEXT NOT NULL,
    off_topic TEXT NOT NULL,
    decision TEXT NOT NULL,
    trim_start_sec REAL,
    trim_end_sec REAL,
    required_repairs_json TEXT,
    normalized_behavior TEXT,
    normalized_norm TEXT,
    rejection_reasons_json TEXT NOT NULL,
    evidence_frames_json TEXT NOT NULL,
    description TEXT NOT NULL,
    raw_response_json TEXT NOT NULL,
    response_id TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    audited_at REAL NOT NULL,
    UNIQUE(item_id, rubric_version, model, pass_index)
);

CREATE INDEX IF NOT EXISTS idx_judgments_lookup
ON judgments(rubric_version, model, decision, item_id);

CREATE TABLE IF NOT EXISTS commentary_judgments (
    judgment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL REFERENCES items(item_id),
    batch_id TEXT REFERENCES batches(batch_id),
    rubric_version TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    content_manifest_sha256 TEXT NOT NULL,
    pass_index INTEGER NOT NULL DEFAULT 0,
    is_social_norm TEXT NOT NULL,
    concrete_behavior TEXT NOT NULL,
    social_actor_grounded TEXT NOT NULL,
    target_or_shared_context_grounded TEXT NOT NULL,
    normative_stance_grounded TEXT NOT NULL,
    proposed_norm_supported TEXT NOT NULL,
    stance_type TEXT NOT NULL,
    quote_relation TEXT NOT NULL,
    decision TEXT NOT NULL,
    rejection_reasons_json TEXT NOT NULL,
    normalized_behavior TEXT NOT NULL,
    normalized_norm TEXT NOT NULL,
    behavior_evidence_quote TEXT NOT NULL,
    stance_evidence_quote TEXT NOT NULL,
    description TEXT NOT NULL,
    raw_response_json TEXT NOT NULL,
    response_id TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    audited_at REAL NOT NULL,
    UNIQUE(item_id, rubric_version, model, pass_index)
);

CREATE INDEX IF NOT EXISTS idx_commentary_judgments_lookup
ON commentary_judgments(rubric_version, model, decision, item_id);

CREATE TABLE IF NOT EXISTS witnessed_judgments (
    judgment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL REFERENCES items(item_id),
    batch_id TEXT REFERENCES batches(batch_id),
    rubric_version TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    frame_manifest_sha256 TEXT NOT NULL,
    pass_index INTEGER NOT NULL DEFAULT 0,
    is_social_norm TEXT NOT NULL,
    social_action_visible TEXT NOT NULL,
    reaction_visible TEXT NOT NULL,
    action_before_reaction TEXT NOT NULL,
    reaction_targets_action TEXT NOT NULL,
    reaction_is_normative TEXT NOT NULL,
    behavior_label_supported TEXT NOT NULL,
    clean_pre_reaction_demo TEXT NOT NULL,
    authenticity TEXT NOT NULL,
    decision TEXT NOT NULL,
    action_end_sec REAL,
    reaction_start_sec REAL,
    rejection_reasons_json TEXT NOT NULL,
    action_evidence_frames_json TEXT NOT NULL,
    reaction_evidence_frames_json TEXT NOT NULL,
    description TEXT NOT NULL,
    raw_response_json TEXT NOT NULL,
    response_id TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    audited_at REAL NOT NULL,
    UNIQUE(item_id, rubric_version, model, pass_index)
);

CREATE INDEX IF NOT EXISTS idx_witnessed_judgments_lookup
ON witnessed_judgments(rubric_version, model,decision,item_id);

CREATE TABLE IF NOT EXISTS instructional_repair_judgments (
    judgment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id TEXT NOT NULL REFERENCES items(item_id),
    batch_id TEXT NOT NULL REFERENCES batches(batch_id),
    rubric_version TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_sha256 TEXT NOT NULL,
    source_frame_manifest_sha256 TEXT NOT NULL,
    repaired_frame_manifest_sha256 TEXT NOT NULL,
    pass_index INTEGER NOT NULL DEFAULT 0,
    repair_type TEXT NOT NULL,
    transform_json TEXT NOT NULL,
    is_social_norm TEXT NOT NULL,
    visual_demo_present TEXT NOT NULL,
    norm_supported TEXT NOT NULL,
    polarity_supported TEXT NOT NULL,
    label_leak_visible TEXT NOT NULL,
    narration_leak TEXT NOT NULL,
    decision TEXT NOT NULL,
    required_repairs_json TEXT NOT NULL,
    normalized_behavior TEXT NOT NULL,
    normalized_norm TEXT NOT NULL,
    rejection_reasons_json TEXT NOT NULL,
    evidence_frames_json TEXT NOT NULL,
    description TEXT NOT NULL,
    raw_response_json TEXT NOT NULL,
    response_id TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    audited_at REAL NOT NULL,
    UNIQUE(batch_id, item_id, model, pass_index)
);

CREATE INDEX IF NOT EXISTS idx_instructional_repair_judgments_lookup
ON instructional_repair_judgments(rubric_version,model,decision,item_id);

CREATE TABLE IF NOT EXISTS rule_evaluations (
    rule_version TEXT NOT NULL,
    item_id TEXT NOT NULL REFERENCES items(item_id),
    judgment_id INTEGER NOT NULL REFERENCES judgments(judgment_id),
    predicted_decision TEXT NOT NULL,
    evaluated_at REAL NOT NULL,
    PRIMARY KEY(rule_version, item_id, judgment_id)
);
"""

ITEM_UPSERT = """
INSERT INTO items(
    item_id,pillar,uid,item_index,metadata_path,clip_path,has_clip,
    title,category,genre,source_platform,found_by_query,query_source,
    polarity,norm,start_quote,end_quote,explanation,start_sec,end_sec,
    present,source_mtime,discovered_at,updated_at
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(item_id) DO UPDATE SET
    metadata_path=excluded.metadata_path,
    clip_path=excluded.clip_path,
    has_clip=excluded.has_clip,
    title=excluded.title,
    category=excluded.category,
    genre=excluded.genre,
    source_platform=excluded.source_platform,
    found_by_query=excluded.found_by_query,
    query_source=excluded.query_source,
    polarity=excluded.polarity,
    norm=excluded.norm,
    start_quote=excluded.start_quote,
    end_quote=excluded.end_quote,
    explanation=excluded.explanation,
    start_sec=excluded.start_sec,
    end_sec=excluded.end_sec,
    present=1,
    source_mtime=excluded.source_mtime,
    updated_at=excluded.updated_at
"""


TRI = {"yes", "no", "uncertain"}
DECISIONS = {"accept", "accept_after_trim", "accept_with_repairs", "reject", "uncertain"}


def is_duplicate_integrity_error(exc: sqlite3.IntegrityError) -> bool:
    """Return true only for the uniqueness collisions used by resumable imports."""
    return "UNIQUE constraint failed" in str(exc)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=60)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    item_columns = {row["name"] for row in conn.execute("PRAGMA table_info(items)")}
    for name, declaration in (
        ("source_platform", "TEXT"),
        ("found_by_query", "TEXT"),
        ("query_source", "TEXT"),
    ):
        if name not in item_columns:
            conn.execute(f"ALTER TABLE items ADD COLUMN {name} {declaration}")
    judgment_columns = {row["name"] for row in conn.execute("PRAGMA table_info(judgments)")}
    for name, declaration in (
        ("dialogue_grounded", "TEXT"),
        ("visual_norm_supported_without_audio", "TEXT"),
        ("clip_composition", "TEXT"),
        ("narration_leak", "TEXT"),
        ("trim_start_sec", "REAL"),
        ("trim_end_sec", "REAL"),
        ("required_repairs_json", "TEXT"),
        ("normalized_behavior", "TEXT"),
        ("normalized_norm", "TEXT"),
    ):
        if name not in judgment_columns:
            conn.execute(f"ALTER TABLE judgments ADD COLUMN {name} {declaration}")
    conn.commit()
    return conn


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def scan_instructional(conn: sqlite3.Connection, project_root: Path) -> dict[str, int]:
    now = time.time()
    root = project_root / "data" / "instructional"
    conn.execute("UPDATE items SET present=0 WHERE pillar='instructional'")
    counts = {"videos": 0, "demos": 0, "clips": 0, "bad_metadata": 0}
    for metadata_path in root.glob("*/metadata.json"):
        metadata = load_json(metadata_path)
        if metadata is None:
            counts["bad_metadata"] += 1
            continue
        counts["videos"] += 1
        uid = str(metadata.get("video_id") or metadata_path.parent.name)
        provenance = metadata.get("provenance") or {}
        category = metadata.get("category") or provenance.get("category")
        source_mtime = metadata_path.stat().st_mtime
        for index, demo in enumerate(metadata.get("demos") or []):
            counts["demos"] += 1
            clip_name = demo.get("clip")
            clip_abs = metadata_path.parent / str(clip_name) if clip_name else None
            has_clip = int(bool(clip_abs and clip_abs.is_file()))
            counts["clips"] += has_clip
            clip_rel = str(clip_abs.relative_to(project_root)) if has_clip and clip_abs else None
            item_id = f"instructional:{uid}:{index}"
            values = (
                item_id,
                "instructional",
                uid,
                index,
                str(metadata_path.relative_to(project_root)),
                clip_rel,
                has_clip,
                metadata.get("title"),
                category,
                metadata.get("genre"),
                metadata.get("source") or provenance.get("platform"),
                provenance.get("found_by_query"),
                provenance.get("query_source"),
                demo.get("polarity"),
                demo.get("norm"),
                demo.get("start_quote"),
                demo.get("end_quote"),
                demo.get("explanation"),
                demo.get("start"),
                demo.get("end"),
                1,
                source_mtime,
                now,
                now,
            )
            conn.execute(ITEM_UPSERT, values)
    conn.commit()
    return counts


def scan_witnessed(conn: sqlite3.Connection, project_root: Path) -> dict[str, int]:
    now = time.time()
    root = project_root / "data" / "hits"
    conn.execute("UPDATE items SET present=0 WHERE pillar='witnessed'")
    counts = {"videos": 0, "reactions": 0, "clips": 0, "bad_metadata": 0}
    for metadata_path in root.glob("*/metadata.json"):
        metadata = load_json(metadata_path)
        if metadata is None:
            counts["bad_metadata"] += 1
            continue
        counts["videos"] += 1
        uid = str(metadata.get("video_id") or metadata_path.parent.name)
        provenance = metadata.get("provenance") or {}
        scene = provenance.get("scene") or {}
        source_mtime = metadata_path.stat().st_mtime
        for ordinal, reaction in enumerate(metadata.get("reactions") or []):
            counts["reactions"] += 1
            clip_index = reaction.get("clip_idx")
            if not isinstance(clip_index, int):
                clip_index = ordinal
            clip_abs = metadata_path.parent / f"clip_{clip_index}.mp4"
            has_clip = int(clip_abs.is_file())
            counts["clips"] += has_clip
            clip_rel = str(clip_abs.relative_to(project_root)) if has_clip else None
            window = reaction.get("clip_window") or [None, None]
            values = (
                f"witnessed:{uid}:{ordinal}",
                "witnessed",
                uid,
                ordinal,
                str(metadata_path.relative_to(project_root)),
                clip_rel,
                has_clip,
                metadata.get("title"),
                metadata.get("category") or provenance.get("category"),
                scene.get("scene_type"),
                metadata.get("source") or provenance.get("platform"),
                provenance.get("found_by_query"),
                provenance.get("query_source"),
                reaction.get("tag"),
                reaction.get("norm"),
                reaction.get("matched_text") or reaction.get("phrase"),
                reaction.get("matched_text") or reaction.get("phrase"),
                reaction.get("context"),
                window[0] if len(window) > 0 else None,
                window[1] if len(window) > 1 else None,
                1,
                source_mtime,
                now,
                now,
            )
            conn.execute(ITEM_UPSERT, values)
    conn.commit()
    return counts


def scan_commentary(conn: sqlite3.Connection, project_root: Path) -> dict[str, int]:
    now = time.time()
    root = project_root / "data" / "discussion"
    conn.execute("UPDATE items SET present=0 WHERE pillar='commentary'")
    counts = {"videos": 0, "statements": 0, "bad_metadata": 0}
    for metadata_path in root.glob("*.json"):
        metadata = load_json(metadata_path)
        if metadata is None:
            counts["bad_metadata"] += 1
            continue
        counts["videos"] += 1
        uid = str(metadata.get("video_id") or metadata_path.stem)
        provenance = metadata.get("provenance") or {}
        scene = provenance.get("scene") or {}
        source_mtime = metadata_path.stat().st_mtime
        for index, statement in enumerate(metadata.get("statements") or []):
            counts["statements"] += 1
            quote = statement.get("quote")
            values = (
                f"commentary:{uid}:{index}",
                "commentary",
                uid,
                index,
                str(metadata_path.relative_to(project_root)),
                None,
                0,
                metadata.get("title"),
                metadata.get("category") or provenance.get("category"),
                scene.get("scene_type"),
                metadata.get("source") or provenance.get("platform"),
                provenance.get("found_by_query"),
                provenance.get("query_source"),
                statement.get("signal"),
                statement.get("norm"),
                quote,
                quote,
                None,
                statement.get("start"),
                statement.get("end"),
                1,
                source_mtime,
                now,
                now,
            )
            conn.execute(ITEM_UPSERT, values)
    conn.commit()
    return counts


def validate_judgment(row: dict[str, Any]) -> None:
    required = {
        "item_id",
        "rubric_version",
        "model",
        "prompt_sha256",
        "frame_manifest_sha256",
        "is_social_norm",
        "visual_demo_present",
        "medium",
        "norm_supported",
        "polarity_supported",
        "explanation_supports_norm",
        "label_leak_visible",
        "off_topic",
        "decision",
        "rejection_reasons",
        "evidence_frames",
        "description",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"missing fields: {missing}")
    for key in (
        "is_social_norm",
        "visual_demo_present",
        "norm_supported",
        "explanation_supports_norm",
        "label_leak_visible",
        "off_topic",
    ):
        if row[key] not in TRI:
            raise ValueError(f"{key} must be yes/no/uncertain")
    if row["decision"] not in DECISIONS:
        raise ValueError("decision must be accept/reject/uncertain")
    if not isinstance(row["rejection_reasons"], list):
        raise ValueError("rejection_reasons must be a list")
    if not isinstance(row["evidence_frames"], list):
        raise ValueError("evidence_frames must be a list")
    if row["decision"] in {"accept", "accept_after_trim"}:
        if row["visual_demo_present"] != "yes" or row["norm_supported"] != "yes":
            raise ValueError("accepted decisions require visual_demo_present=yes and norm_supported=yes")
    if row["decision"] == "accept_after_trim":
        if row.get("label_leak_visible") != "yes" and row.get("narration_leak") != "yes":
            raise ValueError("accept_after_trim requires a visible or narration label leak")
        if row.get("trim_start_sec") is None or row.get("trim_end_sec") is None:
            raise ValueError("accept_after_trim requires trim_start_sec and trim_end_sec")
    if row["rubric_version"] == "instructional_v4":
        v4_required = {
            "visual_norm_supported_without_audio",
            "required_repairs",
            "normalized_behavior",
            "normalized_norm",
        }
        missing_v4 = sorted(v4_required - set(row))
        if missing_v4:
            raise ValueError(f"missing instructional_v4 fields: {missing_v4}")
        if row["visual_norm_supported_without_audio"] not in TRI:
            raise ValueError("visual_norm_supported_without_audio must be yes/no/uncertain")
        if not isinstance(row["required_repairs"], list):
            raise ValueError("required_repairs must be a list")
        accepted_v4 = row["decision"] in {"accept", "accept_with_repairs"}
        if accepted_v4 and (
            row["is_social_norm"] != "yes"
            or row["visual_demo_present"] != "yes"
            or row["polarity_supported"] not in {"violation", "correct", "contrast"}
            or row["off_topic"] != "no"
            or not row["normalized_behavior"].strip()
            or not row["normalized_norm"].strip()
        ):
            raise ValueError("instructional_v4 acceptance violates base invariants")
        if row["decision"] == "accept" and row["required_repairs"]:
            raise ValueError("clean instructional_v4 acceptance cannot require repairs")
        if row["decision"] == "accept_with_repairs" and not row["required_repairs"]:
            raise ValueError("accept_with_repairs requires repairs")
        if not accepted_v4 and row["required_repairs"]:
            raise ValueError("non-accepted instructional_v4 row cannot prescribe repairs")


def import_judgments(conn: sqlite3.Connection, path: Path) -> tuple[int, int]:
    inserted = 0
    skipped = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate_judgment(row)
                conn.execute(
                    """
                    INSERT INTO judgments(
                        item_id,batch_id,rubric_version,model,prompt_sha256,
                        frame_manifest_sha256,pass_index,is_social_norm,
                        visual_demo_present,medium,clip_composition,norm_supported,
                        polarity_supported,explanation_supports_norm,dialogue_grounded,
                        visual_norm_supported_without_audio,narration_leak,label_leak_visible,
                        off_topic,decision,trim_start_sec,trim_end_sec,
                        required_repairs_json,normalized_behavior,normalized_norm,
                        rejection_reasons_json,evidence_frames_json,description,
                        raw_response_json,response_id,input_tokens,output_tokens,audited_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row["item_id"],
                        row.get("batch_id"),
                        row["rubric_version"],
                        row["model"],
                        row["prompt_sha256"],
                        row["frame_manifest_sha256"],
                        int(row.get("pass_index", 0)),
                        row["is_social_norm"],
                        row["visual_demo_present"],
                        row["medium"],
                        row.get("clip_composition"),
                        row["norm_supported"],
                        row["polarity_supported"],
                        row["explanation_supports_norm"],
                        row.get("dialogue_grounded", "uncertain"),
                        row.get("visual_norm_supported_without_audio", "uncertain"),
                        row.get("narration_leak", "uncertain"),
                        row["label_leak_visible"],
                        row["off_topic"],
                        row["decision"],
                        row.get("trim_start_sec"),
                        row.get("trim_end_sec"),
                        json.dumps(row.get("required_repairs", []), ensure_ascii=False),
                        row.get("normalized_behavior", ""),
                        row.get("normalized_norm", ""),
                        json.dumps(row["rejection_reasons"], ensure_ascii=False),
                        json.dumps(row["evidence_frames"]),
                        row["description"],
                        json.dumps(row.get("raw_response", row), ensure_ascii=False),
                        row.get("response_id"),
                        row.get("input_tokens"),
                        row.get("output_tokens"),
                        float(row.get("audited_at", time.time())),
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError as exc:
                if is_duplicate_integrity_error(exc):
                    skipped += 1
                else:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    conn.commit()
    return inserted, skipped


def validate_commentary_judgment(row: dict[str, Any]) -> None:
    required = {
        "item_id",
        "rubric_version",
        "model",
        "prompt_sha256",
        "content_manifest_sha256",
        "is_social_norm",
        "concrete_behavior",
        "social_actor_grounded",
        "target_or_shared_context_grounded",
        "normative_stance_grounded",
        "proposed_norm_supported",
        "stance_type",
        "quote_relation",
        "decision",
        "rejection_reasons",
        "normalized_behavior",
        "normalized_norm",
        "behavior_evidence_quote",
        "stance_evidence_quote",
        "description",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"missing commentary fields: {missing}")
    for key in (
        "is_social_norm",
        "concrete_behavior",
        "social_actor_grounded",
        "target_or_shared_context_grounded",
        "normative_stance_grounded",
        "proposed_norm_supported",
    ):
        if row[key] not in TRI:
            raise ValueError(f"{key} must be yes/no/uncertain")
    if row["decision"] not in {"accept", "accept_after_relabel", "reject", "uncertain"}:
        raise ValueError("invalid commentary decision")
    if not isinstance(row["rejection_reasons"], list):
        raise ValueError("rejection_reasons must be a list")
    if row["decision"] in {"accept", "accept_after_relabel"}:
        required_yes = (
            "is_social_norm",
            "concrete_behavior",
            "social_actor_grounded",
            "target_or_shared_context_grounded",
            "normative_stance_grounded",
        )
        if any(row[key] != "yes" for key in required_yes):
            raise ValueError("accepted commentary row violates grounding invariants")
        if row["stance_type"] in {"none", "uncertain"}:
            raise ValueError("accepted commentary row requires a resolved stance")
        if row["quote_relation"] not in {"same_sentence", "separate_context"}:
            raise ValueError("accepted commentary row requires both behavior and stance")
        if not row["behavior_evidence_quote"].strip() or not row["stance_evidence_quote"].strip():
            raise ValueError("accepted commentary row requires two evidence quotes")
        if not row["normalized_behavior"].strip() or not row["normalized_norm"].strip():
            raise ValueError("accepted commentary row requires normalized behavior and norm")
    if row["decision"] == "accept" and row["proposed_norm_supported"] != "yes":
        raise ValueError("clean commentary acceptance requires the proposed norm")
    if row["decision"] == "accept_after_relabel" and row["proposed_norm_supported"] == "yes":
        raise ValueError("relabeling is reserved for an unsupported proposed norm")


def import_commentary_judgments(conn: sqlite3.Connection, path: Path) -> tuple[int, int]:
    inserted = 0
    skipped = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate_commentary_judgment(row)
                conn.execute(
                    """
                    INSERT INTO commentary_judgments(
                        item_id,batch_id,rubric_version,model,prompt_sha256,
                        content_manifest_sha256,pass_index,is_social_norm,
                        concrete_behavior,social_actor_grounded,
                        target_or_shared_context_grounded,normative_stance_grounded,
                        proposed_norm_supported,stance_type,quote_relation,decision,
                        rejection_reasons_json,normalized_behavior,normalized_norm,
                        behavior_evidence_quote,stance_evidence_quote,description,
                        raw_response_json,response_id,input_tokens,output_tokens,audited_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row["item_id"], row.get("batch_id"), row["rubric_version"],
                        row["model"], row["prompt_sha256"], row["content_manifest_sha256"],
                        int(row.get("pass_index", 0)), row["is_social_norm"],
                        row["concrete_behavior"], row["social_actor_grounded"],
                        row["target_or_shared_context_grounded"],
                        row["normative_stance_grounded"], row["proposed_norm_supported"],
                        row["stance_type"], row["quote_relation"], row["decision"],
                        json.dumps(row["rejection_reasons"], ensure_ascii=False),
                        row["normalized_behavior"], row["normalized_norm"],
                        row["behavior_evidence_quote"], row["stance_evidence_quote"],
                        row["description"],
                        json.dumps(row.get("raw_response", row), ensure_ascii=False),
                        row.get("response_id"), row.get("input_tokens"),
                        row.get("output_tokens"), float(row.get("audited_at", time.time())),
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError as exc:
                if is_duplicate_integrity_error(exc):
                    skipped += 1
                else:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    conn.commit()
    return inserted, skipped


def validate_witnessed_judgment(row: dict[str, Any]) -> None:
    required = {
        "item_id", "rubric_version", "model", "prompt_sha256",
        "frame_manifest_sha256", "is_social_norm", "social_action_visible",
        "reaction_visible", "action_before_reaction", "reaction_targets_action",
        "reaction_is_normative", "behavior_label_supported",
        "clean_pre_reaction_demo", "authenticity", "decision", "action_end_sec",
        "reaction_start_sec", "rejection_reasons", "action_evidence_frames",
        "reaction_evidence_frames", "description",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"missing witnessed fields: {missing}")
    tri_fields = (
        "is_social_norm", "social_action_visible", "reaction_visible",
        "action_before_reaction", "reaction_targets_action", "reaction_is_normative",
        "behavior_label_supported", "clean_pre_reaction_demo",
    )
    for key in tri_fields:
        if row[key] not in TRI:
            raise ValueError(f"{key} must be yes/no/uncertain")
    if row["decision"] not in {
        "accept_after_splice", "reroute_instructional", "reject", "uncertain"
    }:
        raise ValueError("invalid witnessed decision")
    for key in ("rejection_reasons", "action_evidence_frames", "reaction_evidence_frames"):
        if not isinstance(row[key], list):
            raise ValueError(f"{key} must be a list")
    if row["decision"] == "accept_after_splice":
        if any(row[key] != "yes" for key in tri_fields):
            raise ValueError("witnessed acceptance violates sequence invariants")
        if row["authenticity"] not in {"organic", "hidden_camera_genuine"}:
            raise ValueError("witnessed acceptance requires organic authenticity")
        if row["action_end_sec"] is None or row["reaction_start_sec"] is None:
            raise ValueError("witnessed acceptance requires splice bounds")
        if not 0 < float(row["action_end_sec"]) <= float(row["reaction_start_sec"]):
            raise ValueError("invalid witnessed splice ordering")
    if row["decision"] == "reroute_instructional":
        if row["social_action_visible"] != "yes" or row["behavior_label_supported"] != "yes":
            raise ValueError("instructional reroute requires a visible labeled behavior")
        if row["authenticity"] not in {"scripted", "animation"}:
            raise ValueError("instructional reroute requires scripted media")


def import_witnessed_judgments(conn: sqlite3.Connection, path: Path) -> tuple[int, int]:
    inserted = 0
    skipped = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate_witnessed_judgment(row)
                conn.execute(
                    """
                    INSERT INTO witnessed_judgments(
                        item_id,batch_id,rubric_version,model,prompt_sha256,
                        frame_manifest_sha256,pass_index,is_social_norm,
                        social_action_visible,reaction_visible,action_before_reaction,
                        reaction_targets_action,reaction_is_normative,behavior_label_supported,
                        clean_pre_reaction_demo,authenticity,decision,action_end_sec,
                        reaction_start_sec,rejection_reasons_json,
                        action_evidence_frames_json,reaction_evidence_frames_json,description,
                        raw_response_json,response_id,input_tokens,output_tokens,audited_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row["item_id"], row.get("batch_id"), row["rubric_version"],
                        row["model"], row["prompt_sha256"], row["frame_manifest_sha256"],
                        int(row.get("pass_index", 0)), row["is_social_norm"],
                        row["social_action_visible"], row["reaction_visible"],
                        row["action_before_reaction"], row["reaction_targets_action"],
                        row["reaction_is_normative"], row["behavior_label_supported"],
                        row["clean_pre_reaction_demo"], row["authenticity"], row["decision"],
                        row["action_end_sec"], row["reaction_start_sec"],
                        json.dumps(row["rejection_reasons"], ensure_ascii=False),
                        json.dumps(row["action_evidence_frames"]),
                        json.dumps(row["reaction_evidence_frames"]), row["description"],
                        json.dumps(row.get("raw_response", row), ensure_ascii=False),
                        row.get("response_id"), row.get("input_tokens"),
                        row.get("output_tokens"), float(row.get("audited_at", time.time())),
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError as exc:
                if is_duplicate_integrity_error(exc):
                    skipped += 1
                else:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    conn.commit()
    return inserted, skipped


def validate_instructional_repair_judgment(row: dict[str, Any]) -> None:
    required = {
        "item_id", "batch_id", "rubric_version", "model", "prompt_sha256",
        "source_frame_manifest_sha256", "repaired_frame_manifest_sha256",
        "repair_type", "transform", "is_social_norm", "visual_demo_present",
        "norm_supported", "polarity_supported", "label_leak_visible",
        "narration_leak", "decision", "required_repairs",
        "normalized_behavior", "normalized_norm", "rejection_reasons",
        "evidence_frames", "description",
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"missing instructional repair fields: {missing}")
    for key in (
        "is_social_norm", "visual_demo_present", "norm_supported",
        "label_leak_visible", "narration_leak",
    ):
        if row[key] not in TRI:
            raise ValueError(f"{key} must be yes/no/uncertain")
    if row["repair_type"] not in {
        "tighten_to_demo", "crop_label_overlay", "strip_explanatory_audio"
    }:
        raise ValueError("invalid instructional repair_type")
    if row["decision"] not in {"accept", "accept_after_relabel", "reject", "uncertain"}:
        raise ValueError("invalid instructional repair decision")
    if not isinstance(row["transform"], dict):
        raise ValueError("transform must be an object")
    for key in ("required_repairs", "rejection_reasons", "evidence_frames"):
        if not isinstance(row[key], list):
            raise ValueError(f"{key} must be a list")
    if row["repair_type"] == "tighten_to_demo":
        start = row["transform"].get("start_sec")
        end = row["transform"].get("end_sec")
        source_duration = row["transform"].get("source_duration_sec")
        if start is None or end is None or source_duration is None:
            raise ValueError("tighten_to_demo requires start/end/source duration")
        if not 0 <= float(start) < float(end) <= float(source_duration):
            raise ValueError("invalid tighten_to_demo bounds")
    elif row["repair_type"] == "strip_explanatory_audio":
        before = row["transform"].get("source_audio_stream_count")
        after = row["transform"].get("repaired_audio_stream_count")
        if row["transform"].get("audio_removed") is not True:
            raise ValueError("strip_explanatory_audio requires audio_removed=true")
        if before is None or int(before) < 1 or after is None or int(after) != 0:
            raise ValueError("strip_explanatory_audio requires verified stream counts")
    elif row["repair_type"] == "crop_label_overlay":
        if not str(row["transform"].get("crop_geometry") or "").strip():
            raise ValueError("crop_label_overlay requires crop_geometry")
    accepted = row["decision"] in {"accept", "accept_after_relabel"}
    if accepted and (
        row["is_social_norm"] != "yes"
        or row["visual_demo_present"] != "yes"
        or row["polarity_supported"] not in {"violation", "correct", "contrast"}
        or row["label_leak_visible"] != "no"
        or row["narration_leak"] != "no"
        or not row["normalized_behavior"].strip()
        or not row["normalized_norm"].strip()
    ):
        raise ValueError("instructional repair acceptance violates evidence invariants")
    if row["decision"] == "accept":
        if row["norm_supported"] != "yes" or row["required_repairs"]:
            raise ValueError("clean repaired acceptance requires supported norm and no repairs")
    if row["decision"] == "accept_after_relabel":
        relabels = set(row["required_repairs"])
        if not relabels.issubset({"relabel_norm", "relabel_polarity"}):
            raise ValueError("post-repair relabel may only change norm or polarity")
        if not relabels:
            raise ValueError("relabel acceptance requires a relabel repair")
        if "relabel_norm" in relabels and row["norm_supported"] != "no":
            raise ValueError("norm relabel requires norm_supported=no")
        if "relabel_norm" not in relabels and row["norm_supported"] != "yes":
            raise ValueError("polarity-only relabel requires norm_supported=yes")
    if not accepted and row["required_repairs"]:
        raise ValueError("non-accepted repair judgment cannot prescribe repairs")


def import_instructional_repair_judgments(
    conn: sqlite3.Connection, path: Path
) -> tuple[int, int]:
    inserted = 0
    skipped = 0
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                validate_instructional_repair_judgment(row)
                conn.execute(
                    """
                    INSERT INTO instructional_repair_judgments(
                        item_id,batch_id,rubric_version,model,prompt_sha256,
                        source_frame_manifest_sha256,repaired_frame_manifest_sha256,
                        pass_index,repair_type,transform_json,is_social_norm,
                        visual_demo_present,norm_supported,polarity_supported,
                        label_leak_visible,narration_leak,decision,
                        required_repairs_json,normalized_behavior,normalized_norm,
                        rejection_reasons_json,evidence_frames_json,description,
                        raw_response_json,response_id,input_tokens,output_tokens,audited_at
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        row["item_id"], row["batch_id"], row["rubric_version"],
                        row["model"], row["prompt_sha256"],
                        row["source_frame_manifest_sha256"],
                        row["repaired_frame_manifest_sha256"],
                        int(row.get("pass_index", 0)), row["repair_type"],
                        json.dumps(row["transform"], ensure_ascii=False),
                        row["is_social_norm"], row["visual_demo_present"],
                        row["norm_supported"], row["polarity_supported"],
                        row["label_leak_visible"], row["narration_leak"],
                        row["decision"],
                        json.dumps(row["required_repairs"], ensure_ascii=False),
                        row["normalized_behavior"], row["normalized_norm"],
                        json.dumps(row["rejection_reasons"], ensure_ascii=False),
                        json.dumps(row["evidence_frames"]), row["description"],
                        json.dumps(row.get("raw_response", row), ensure_ascii=False),
                        row.get("response_id"), row.get("input_tokens"),
                        row.get("output_tokens"), float(row.get("audited_at", time.time())),
                    ),
                )
                inserted += 1
            except sqlite3.IntegrityError as exc:
                if is_duplicate_integrity_error(exc):
                    skipped += 1
                else:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: {exc}") from exc
    conn.commit()
    return inserted, skipped


def print_stats(conn: sqlite3.Connection) -> None:
    for row in conn.execute(
        "SELECT pillar,present,has_clip,COUNT(*) n FROM items GROUP BY pillar,present,has_clip"
    ):
        print(dict(row))
    print("instructional_repair_judgments")
    for row in conn.execute(
        """
        SELECT rubric_version,model,repair_type,decision,COUNT(*) n
        FROM instructional_repair_judgments
        GROUP BY rubric_version,model,repair_type,decision
        ORDER BY rubric_version,model,repair_type,decision
        """
    ):
        print(dict(row))
    print("witnessed_judgments")
    for row in conn.execute(
        """
        SELECT rubric_version,model,decision,COUNT(*) n
        FROM witnessed_judgments GROUP BY rubric_version,model,decision
        ORDER BY rubric_version,model,decision
        """
    ):
        print(dict(row))
    print("commentary_judgments")
    for row in conn.execute(
        """
        SELECT rubric_version,model,decision,COUNT(*) n
        FROM commentary_judgments GROUP BY rubric_version,model,decision
        ORDER BY rubric_version,model,decision
        """
    ):
        print(dict(row))
    print("judgments")
    for row in conn.execute(
        """
        SELECT rubric_version,model,decision,COUNT(*) n
        FROM judgments GROUP BY rubric_version,model,decision
        ORDER BY rubric_version,model,decision
        """
    ):
        print(dict(row))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/visual_audit/audit.db"))
    sub = parser.add_subparsers(dest="command", required=True)
    scan = sub.add_parser("scan")
    scan.add_argument("--project-root", type=Path, default=Path("."))
    scan.add_argument(
        "--pillar",
        choices=("all", "instructional", "witnessed", "commentary"),
        default="all",
    )
    imp = sub.add_parser("import")
    imp.add_argument("results", type=Path)
    imp_commentary = sub.add_parser("import-commentary")
    imp_commentary.add_argument("results", type=Path)
    imp_witnessed = sub.add_parser("import-witnessed")
    imp_witnessed.add_argument("results", type=Path)
    imp_repair = sub.add_parser("import-instructional-repairs")
    imp_repair.add_argument("results", type=Path)
    sub.add_parser("stats")
    args = parser.parse_args()

    conn = connect(args.db)
    if args.command == "scan":
        project_root = args.project_root.resolve()
        scanners = {
            "instructional": scan_instructional,
            "witnessed": scan_witnessed,
            "commentary": scan_commentary,
        }
        selected = scanners if args.pillar == "all" else {args.pillar: scanners[args.pillar]}
        counts = {name: scanner(conn, project_root) for name, scanner in selected.items()}
        print(json.dumps(counts, sort_keys=True))
    elif args.command == "import":
        inserted, skipped = import_judgments(conn, args.results)
        print(json.dumps({"inserted": inserted, "skipped": skipped}))
    elif args.command == "import-commentary":
        inserted, skipped = import_commentary_judgments(conn, args.results)
        print(json.dumps({"inserted": inserted, "skipped": skipped}))
    elif args.command == "import-witnessed":
        inserted, skipped = import_witnessed_judgments(conn, args.results)
        print(json.dumps({"inserted": inserted, "skipped": skipped}))
    elif args.command == "import-instructional-repairs":
        inserted, skipped = import_instructional_repair_judgments(conn, args.results)
        print(json.dumps({"inserted": inserted, "skipped": skipped}))
    else:
        print_stats(conn)
    conn.close()


if __name__ == "__main__":
    main()
