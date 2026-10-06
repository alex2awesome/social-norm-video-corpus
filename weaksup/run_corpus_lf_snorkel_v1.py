#!/usr/bin/env python3
"""Corpus-wide LF materialization + shadow label-model run.

Walks the live corpus layout (data/hits, data/instructional, data/discussion,
data/transcripts), materializes standardized LF records and eligibility gates
per item, fits the family label model per (pillar, target), and exports
append-only shadow posteriors plus a ranked witnessed norm-violation review
list.  Resumable via an append-only per-source journal.  CPU-only; touches no
GPU, mutates no corpus file, deletes nothing, and emits no acceptance label.

Typical launch on sk3:

    nohup <env-python> -u scripts/run_corpus_lf_snorkel_v1.py \
        --root /lfs/skampere3/0/alexspan/norm-scraper \
        --out-dir /lfs/skampere3/0/alexspan/norm-scraper/data/shadow_scores/20260817_lf_snorkel_v1 \
        > logs/lf_snorkel_v1.log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

try:
    from weaksup.label_model_v1 import score_matrix
    from weaksup.lf_matrix_v1 import build_matrices, iter_jsonl
    from weaksup.materialize_corpus_lf_records_v1 import (
        MATERIALIZER_VERSION,
        commentary_item_records,
        instructional_item_records,
        witnessed_item_records,
    )
    from weaksup.weak_signal_registry import load_registry, validate_registry
    from weaksup.visual_feature_lfs_v1 import (
        commentary_visual_lf_records,
        instructional_visual_lf_records,
        load_feature_index,
        witnessed_visual_lf_records,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from weaksup.label_model_v1 import score_matrix
    from weaksup.lf_matrix_v1 import build_matrices, iter_jsonl
    from weaksup.materialize_corpus_lf_records_v1 import (
        MATERIALIZER_VERSION,
        commentary_item_records,
        instructional_item_records,
        witnessed_item_records,
    )
    from weaksup.weak_signal_registry import load_registry, validate_registry
    from weaksup.visual_feature_lfs_v1 import (
        commentary_visual_lf_records,
        instructional_visual_lf_records,
        load_feature_index,
        witnessed_visual_lf_records,
    )


RUN_VERSION = "run_corpus_lf_snorkel_v1"

MODEL_TARGETS = (
    ("witnessed", "norm_event_supported"),
    ("witnessed", "reaction_grounded"),
    ("witnessed", "independent_bystander_signal"),
    ("instructional", "demonstration_present"),
    ("commentary", "occurred_event_supported"),
    ("commentary", "event_present_in_source"),
)


def load_visual_features(features_dir: Path | None) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Join the 20260724 cheap-feature ledgers (pose/CLIP for witnessed and
    instructional; low-level only for commentary, honoring its recorded
    pose/CLIP exclusion)."""
    if features_dir is None:
        return {}
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    for name in (
        "multimodal_v1/witnessed_pose_clip.jsonl",
        "multimodal_v1/instructional_pose_clip.jsonl",
        "commentary_low_level_canonical.jsonl",
    ):
        index.update(load_feature_index(features_dir / name))
    # Later repair passes over previously errored media (append-only shards).
    for rerun in sorted(features_dir.glob("commentary_low_level_rerun*.jsonl")):
        index.update(load_feature_index(rerun))
    return index


def _append_jsonl(handle, rows: Iterable[dict[str, Any]]) -> int:
    n = 0
    for row in rows:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        n += 1
    return n


def _load_transcript_segments(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    segments = value.get("segments") or []
    return segments if isinstance(segments, list) else []


def _augment_with_visual(
    pillar: str,
    uid: str,
    records: list[dict[str, Any]],
    gates: list[dict[str, Any]],
    visual_index: dict[tuple[str, str, int], dict[str, Any]],
) -> None:
    if not visual_index:
        return
    builders = {
        "witnessed": witnessed_visual_lf_records,
        "instructional": instructional_visual_lf_records,
        "commentary": commentary_visual_lf_records,
    }
    for gate in gates:
        item_id = gate["item_id"]
        tail = item_id.rsplit("_", 1)[-1]
        idx = int(tail) if tail.isdigit() else 0
        features = visual_index.get((pillar, uid, 0 if pillar == "commentary" else idx))
        if features:
            records.extend(builders[pillar](item_id, features))


def materialize(root: Path, out_dir: Path, registry: dict[str, Any],
                *, limit: int | None = None,
                visual_index: dict[tuple[str, str, int], dict[str, Any]] | None = None,
                dual_vlm_index: dict[str, bool] | None = None,
                ) -> dict[str, Any]:
    visual_index = visual_index or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = out_dir / "materialize_journal.jsonl"
    done: set[str] = set()
    if journal_path.exists():
        for row in iter_jsonl(journal_path):
            done.add(row["source_key"])

    counts = {"sources": 0, "skipped": 0, "errors": 0, "lf_records": 0, "gates": 0}
    started = time.time()
    with (out_dir / "lf_records.jsonl").open("a") as records_out, \
            (out_dir / "gates.jsonl").open("a") as gates_out, \
            (out_dir / "materialize_errors.jsonl").open("a") as errors_out, \
            journal_path.open("a") as journal:

        def process(source_key: str, fn) -> bool:
            if source_key in done:
                counts["skipped"] += 1
                return True
            try:
                records, gates = fn()
            except Exception as error:  # keep the sweep alive; log and continue
                counts["errors"] += 1
                errors_out.write(json.dumps({
                    "source_key": source_key,
                    "error": f"{type(error).__name__}: {error}",
                    "trace": traceback.format_exc(limit=3),
                }, sort_keys=True) + "\n")
                return True
            counts["lf_records"] += _append_jsonl(records_out, records)
            counts["gates"] += _append_jsonl(gates_out, gates)
            journal.write(json.dumps(
                {"source_key": source_key, "n_records": len(records)},
                sort_keys=True) + "\n")
            counts["sources"] += 1
            if counts["sources"] % 2000 == 0:
                for handle in (records_out, gates_out, journal):
                    handle.flush()
                print(f"[{time.time() - started:8.0f}s] {counts}", flush=True)
            return limit is None or counts["sources"] < limit

        # Witnessed: data/hits/{uid}/metadata.json + transcript.
        for metadata_path in sorted((root / "data" / "hits").glob("*/metadata.json")):
            uid = metadata_path.parent.name

            def witnessed(metadata_path=metadata_path, uid=uid):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                segments = _load_transcript_segments(
                    root / "data" / "transcripts" / f"{uid}.json"
                )
                clip_exists = {}
                for clip_path in metadata_path.parent.glob("clip_*.mp4"):
                    stem = clip_path.stem.rsplit("_", 1)[-1]
                    if stem.isdigit():
                        clip_exists[int(stem)] = clip_path.stat().st_size > 0
                records, gates = witnessed_item_records(
                    uid, metadata, segments, registry, clip_exists=clip_exists
                )
                _augment_with_visual("witnessed", uid, records, gates, visual_index)
                return records, gates

            if not process(f"witnessed:{uid}", witnessed):
                break

        # Instructional: data/instructional/{uid}/metadata.json.
        for metadata_path in sorted((root / "data" / "instructional").glob("*/metadata.json")):
            uid = metadata_path.parent.name

            def instructional(metadata_path=metadata_path, uid=uid):
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                demo_exists = {}
                for demo_path in metadata_path.parent.glob("demo_*.mp4"):
                    stem = demo_path.stem.rsplit("_", 1)[-1]
                    if stem.isdigit():
                        demo_exists[int(stem)] = demo_path.stat().st_size > 0
                records, gates = instructional_item_records(
                    uid, metadata, registry, demo_clip_exists=demo_exists
                )
                _augment_with_visual("instructional", uid, records, gates, visual_index)
                return records, gates

            if not process(f"instructional:{uid}", instructional):
                break

        # Commentary: data/discussion/{uid}.json.
        for record_path in sorted((root / "data" / "discussion").glob("*.json")):
            uid = record_path.stem

            def commentary(record_path=record_path, uid=uid):
                record = json.loads(record_path.read_text(encoding="utf-8"))
                segments = _load_transcript_segments(
                    root / "data" / "transcripts" / f"{uid}.json"
                )
                records, gates = commentary_item_records(
                    uid, record, registry,
                    transcript_segments=segments,
                    dual_vlm_positive=(dual_vlm_index or {}).get(uid),
                )
                _augment_with_visual("commentary", uid, records, gates, visual_index)
                return records, gates

            if not process(f"commentary:{uid}", commentary):
                break

    counts["elapsed_sec"] = round(time.time() - started, 1)
    return counts


def fit_and_export(out_dir: Path, *, high: float, low: float) -> dict[str, Any]:
    model_dir = out_dir / "model"
    model_dir.mkdir(exist_ok=True)
    gates: dict[str, dict[str, Any]] = {}
    for row in iter_jsonl(out_dir / "gates.jsonl"):
        gates[row["item_id"]] = {
            "eligible": row.get("eligible"),
            "failed_gates": row.get("failed_gates") or [],
            "unknown_gates": row.get("unknown_gates") or [],
        }
    matrices = build_matrices(iter_jsonl(out_dir / "lf_records.jsonl"))
    summaries = {}
    posteriors_by_item: dict[str, dict[str, float]] = {}
    for pillar, target in MODEL_TARGETS:
        key = (pillar, target)
        if key not in matrices:
            summaries[f"{pillar}:{target}"] = {"rows": 0, "note": "no records"}
            continue
        model, rows = score_matrix(
            matrices[key], pillar=pillar, target=target, gates=gates,
            high=high, low=low,
        )
        out_path = model_dir / f"posteriors_{pillar}_{target}.jsonl"
        with out_path.open("w") as handle:
            _append_jsonl(handle, rows)
        bands: dict[str, int] = {}
        for row in rows:
            bands[row["shadow_band"]] = bands.get(row["shadow_band"], 0) + 1
            posteriors_by_item.setdefault(row["item_id"], {})[target] = row[
                "posterior_positive"
            ]
        summaries[f"{pillar}:{target}"] = {
            "rows": len(rows),
            "bands": bands,
            "model": model.parameters(),
        }
    return {"summaries": summaries, "posteriors_by_item": posteriors_by_item}


def export_ranked_candidates(out_dir: Path, root: Path,
                             posteriors_by_item: dict[str, dict[str, float]],
                             *, high: float) -> dict[str, Any]:
    """Ranked witnessed review list: high-posterior norm events, ordered by
    posterior evidence x detector severity/strength.  Review priority only."""
    ranked = []
    for item_id, targets in posteriors_by_item.items():
        if not item_id.startswith("witnessed:"):
            continue
        norm_p = targets.get("norm_event_supported")
        reaction_p = targets.get("reaction_grounded")
        if norm_p is None or norm_p < high:
            continue
        uid = item_id.split(":")[1]
        severity = strength = None
        metadata_path = root / "data" / "hits" / uid / "metadata.json"
        if metadata_path.is_file():
            try:
                scene = (json.loads(metadata_path.read_text()).get("provenance") or {}).get("scene") or {}
                severity = scene.get("severity")
                strength = scene.get("reaction_strength")
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
        score = norm_p * (reaction_p if reaction_p is not None else 0.5)
        if isinstance(severity, (int, float)) and isinstance(strength, (int, float)):
            score *= (float(severity) * float(strength)) / 25.0
        ranked.append({
            "item_id": item_id,
            "uid": uid,
            "posterior_norm_event_supported": norm_p,
            "posterior_reaction_grounded": reaction_p,
            "posterior_independent_bystander_signal": targets.get(
                "independent_bystander_signal"
            ),
            "severity": severity,
            "reaction_strength": strength,
            "review_priority_score": score,
            "acceptance_label": None,
            "corpus_disposition": None,
            "delete_media": False,
            "policy": "ranked_manual_review_candidates_only",
        })
    ranked.sort(key=lambda row: -row["review_priority_score"])
    out_path = out_dir / "ranked_norm_violation_candidates.jsonl"
    with out_path.open("w") as handle:
        _append_jsonl(handle, ranked)
    return {"ranked_candidates": len(ranked), "output": str(out_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after N sources (smoke runs)")
    parser.add_argument("--visual-features-dir", type=Path, default=None,
                        help="20260724 full-corpus shadow feature dir; adds visual LFs")
    parser.add_argument("--dual-vlm-scores", type=Path, default=None,
                        help="JSONL of {uid, positive} dual-VLM retrieval scores; "
                             "sources without a row abstain")
    parser.add_argument("--high", type=float, default=0.85)
    parser.add_argument("--low", type=float, default=0.15)
    parser.add_argument("--skip-materialize", action="store_true")
    args = parser.parse_args()
    registry_path = args.registry or args.root / "config" / "audited_weak_signals_v1.json"
    registry = load_registry(registry_path)
    validate_registry(registry)  # structural check; artifact hashes checked in CI

    summary: dict[str, Any] = {
        "run_version": RUN_VERSION,
        "materializer_version": MATERIALIZER_VERSION,
        "root": str(args.root),
        "high_band": args.high,
        "low_band": args.low,
    }
    if not args.skip_materialize:
        visual_index = load_visual_features(args.visual_features_dir)
        summary["visual_feature_rows"] = len(visual_index)
        dual_vlm_index: dict[str, bool] = {}
        if args.dual_vlm_scores:
            for row in iter_jsonl(args.dual_vlm_scores):
                dual_vlm_index[row["uid"]] = bool(row["positive"])
        summary["dual_vlm_scored_sources"] = len(dual_vlm_index)
        summary["materialize"] = materialize(
            args.root, args.out_dir, registry, limit=args.limit,
            visual_index=visual_index, dual_vlm_index=dual_vlm_index,
        )
        print(json.dumps({"materialize": summary["materialize"]}, sort_keys=True), flush=True)
    fitted = fit_and_export(args.out_dir, high=args.high, low=args.low)
    summary["models"] = fitted["summaries"]
    summary["ranked_export"] = export_ranked_candidates(
        args.out_dir, args.root, fitted["posteriors_by_item"], high=args.high
    )
    summary["policy"] = "shadow_only_no_acceptance_no_deletion"
    (args.out_dir / "RUN_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
