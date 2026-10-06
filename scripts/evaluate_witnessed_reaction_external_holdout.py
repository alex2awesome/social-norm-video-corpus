#!/usr/bin/env python3
"""Evaluate witnessed-reaction weak signals on a source-disjoint video audit.

The 24-item holdout was manually reviewed before this evaluator was written.
Manual fields are used only to derive evaluation labels and error descriptions;
they are never exposed to labeling functions or fitted models.

This script is audit-only and never mutates the corpus.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from scripts.evaluate_witnessed_reaction_features import (
        DIRECT_ADDRESS,
        DIRECT_OBJECTION,
        GENERIC_AFFECT,
        PROTECTIVE,
        REPORTED,
        TOKEN,
        build_rows as build_training_rows,
        load_jsonl,
        reaction_segment,
        vlm_features,
    )
    from scripts.witnessed_reaction_candidate_scan import (
        candidate_scan_features,
        scan_reaction_candidates,
        serialize_candidates,
    )
    from scripts.witnessed_reaction_identity_features import (
        acoustic_identity_features,
        face_identity_features,
        intervention_features,
        role_separation_features,
        speaker_turn_features,
        temporal_structure_features,
    )
else:
    from evaluate_witnessed_reaction_features import (
        DIRECT_ADDRESS,
        DIRECT_OBJECTION,
        GENERIC_AFFECT,
        PROTECTIVE,
        REPORTED,
        TOKEN,
        build_rows as build_training_rows,
        load_jsonl,
        reaction_segment,
        vlm_features,
    )
    from witnessed_reaction_candidate_scan import (
        candidate_scan_features,
        scan_reaction_candidates,
        serialize_candidates,
    )
    from witnessed_reaction_identity_features import (
        acoustic_identity_features,
        face_identity_features,
        intervention_features,
        role_separation_features,
        speaker_turn_features,
        temporal_structure_features,
    )


STRICT_ROLES = {"bystander", "organic_audience"}
EXTENDED_ROLES = STRICT_ROLES | {"authority_or_host"}
TARGETED_CONTENT = {
    "targeted_objection",
    "correction_or_sanction",
    "protective_intervention",
}
VALID_TEMPORAL = {
    "action_established_before_reaction",
    "action_only_overlaps_reaction",
}
VALID_GROUNDING = {"visible_on_scene", "audible_on_scene"}


def manual_target(row: dict[str, Any], *, include_authority: bool) -> bool:
    """Derive a reaction-only gold label from blinded manual atomic fields."""
    roles = EXTENDED_ROLES if include_authority else STRICT_ROLES
    return (
        row.get("reaction_source_role") in roles
        and row.get("reaction_grounding") in VALID_GROUNDING
        and row.get("reaction_content") in TARGETED_CONTENT
        and row.get("temporal_relation") in VALID_TEMPORAL
    )


def vlm_label(result: dict[str, Any], *, include_authority: bool) -> int:
    """Frozen all-atom VLM labeling function."""
    roles = EXTENDED_ROLES if include_authority else STRICT_ROLES
    return int(
        result.get("reaction_visible_or_audibly_grounded") == "yes"
        and result.get("reaction_source_role") in roles
        and result.get("reaction_content") in TARGETED_CONTENT
        and result.get("reaction_targets_action") == "yes"
        and result.get("action_established_before_reaction") == "yes"
    )


def text_label(reaction: str, context: str, title: str) -> int:
    """High-precision lexical intervention LF; identity is deliberately absent."""
    text = f"{reaction} {context}".strip()
    negative = (
        bool(GENERIC_AFFECT.fullmatch(reaction.strip()))
        or bool(REPORTED.search(f"{title} {text}"))
    )
    active = bool(
        DIRECT_OBJECTION.search(text)
        or PROTECTIVE.search(text)
        or intervention_features(reaction)["intervention.any_active"]
    )
    return int(active and not negative)


def metric_record(gold: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    pred = score >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold, pred, average="binary", zero_division=0
    )
    return {
        "average_precision": float(average_precision_score(gold, score)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "predicted_positive": int(pred.sum()),
        "true_positive": int(((pred == 1) & (gold == 1)).sum()),
        "false_positive": int(((pred == 1) & (gold == 0)).sum()),
        "false_negative": int(((pred == 0) & (gold == 1)).sum()),
    }


def _base_text_features(
    source: dict[str, Any],
    segments: list[dict[str, Any]],
    reaction_index: int | None,
    alignment: float,
) -> dict[str, float]:
    reaction = str(source.get("start_quote") or "")
    context = str(source.get("explanation") or "")
    title = str(source.get("title") or "")
    segment = segments[reaction_index] if reaction_index is not None else {}
    segment_text = str(segment.get("text") or "")
    segment_start = (
        float(segment["clip_start"])
        if segment.get("clip_start") is not None
        else None
    )
    transcript_text = " ".join(str(s.get("text") or "") for s in segments)
    words = TOKEN.findall(reaction.lower())
    text = f"{reaction} {context}"
    return {
        "text.direct_objection": float(bool(DIRECT_OBJECTION.search(text))),
        "text.protective": float(bool(PROTECTIVE.search(text))),
        "text.direct_address": float(bool(DIRECT_ADDRESS.search(text))),
        "text.generic_affect_only": float(bool(GENERIC_AFFECT.match(reaction))),
        "text.reported": float(bool(REPORTED.search(f"{title} {transcript_text}"))),
        "text.reaction_words": float(len(words)),
        "text.negation_count": float(
            sum(word in {"no", "not", "don't", "stop", "never"} for word in words)
        ),
        "text.alignment": float(alignment),
        "text.segment_short": float(0 < len(TOKEN.findall(segment_text)) <= 8),
        "text.has_prior_context": float(
            segment_start is not None and segment_start >= 1.0
        ),
    }


def build_holdout_rows(
    root: Path,
    *,
    extract_media_features: bool = True,
) -> list[dict[str, Any]]:
    source_by = {
        row["item_id"]: row
        for row in json.loads((root / "manifest.json").read_text())["items"]
    }
    benchmark_by = {
        row["item_id"]: row
        for row in load_jsonl(root / "video_benchmark_v1/manifest_v7.jsonl")
    }
    manual_by = {
        row["item_id"]: row for row in load_jsonl(root / "manual_v2_review.jsonl")
    }
    qwen_by = {
        row["item_id"]: row
        for row in load_jsonl(root / "video_benchmark_v1/qwen_v7b_atomic.jsonl")
    }
    glm_by = {
        row["item_id"]: row
        for row in load_jsonl(root / "video_benchmark_v1/glm_v7b_atomic.jsonl")
    }
    common = set.intersection(
        *map(set, (source_by, benchmark_by, manual_by, qwen_by, glm_by))
    )
    rows = []
    for item_id in sorted(common):
        source, benchmark = source_by[item_id], benchmark_by[item_id]
        qwen_result = qwen_by[item_id].get("result") or {}
        glm_result = glm_by[item_id].get("result") or {}
        reaction = str(source.get("start_quote") or "")
        context = str(source.get("explanation") or "")
        segments = benchmark.get("aligned_transcript") or []
        segment, alignment = reaction_segment(reaction, context, segments)
        reaction_index = segments.index(segment) if segment is not None else None
        boundary = (
            float(segment["clip_start"])
            if segment is not None and segment.get("clip_start") is not None
            else None
        )
        features = _base_text_features(
            source, segments, reaction_index, alignment
        )
        features.update(vlm_features("qwen", qwen_by[item_id]))
        features.update(vlm_features("glm", glm_by[item_id]))
        features.update(intervention_features(reaction))
        action_speaker = (
            segments[reaction_index - 1].get("speaker")
            if reaction_index is not None and reaction_index > 0
            else None
        )
        features.update(
            speaker_turn_features(
                segments, reaction_index, action_speaker=action_speaker
            )
        )
        features.update(temporal_structure_features(segments, reaction_index))
        candidates = scan_reaction_candidates(
            segments,
            selected_quote=reaction,
        )
        features.update(
            candidate_scan_features(
                candidates,
                selected_boundary=boundary,
            )
        )
        features.update(
            role_separation_features(
                source.get("violator_role"),
                source.get("reactor_role"),
                vlm_roles=[
                    qwen_result.get("reaction_source_role"),
                    glm_result.get("reaction_source_role"),
                ],
            )
        )
        clip = Path(str(benchmark["proxy_clip"]))
        if extract_media_features:
            features.update(acoustic_identity_features(clip, boundary))
            features.update(face_identity_features(clip, boundary))
        rows.append(
            {
                "item_id": item_id,
                "uid": source["uid"],
                "reaction": reaction,
                "context": context,
                "title": str(source.get("title") or ""),
                "manual": manual_by[item_id],
                "qwen_result": qwen_result,
                "glm_result": glm_result,
                "candidates": serialize_candidates(candidates),
                "features": features,
            }
        )
    return rows


def evaluate_frozen_rules(
    rows: list[dict[str, Any]], *, include_authority: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    gold = np.asarray(
        [
            manual_target(row["manual"], include_authority=include_authority)
            for row in rows
        ],
        dtype=int,
    )
    qwen = np.asarray(
        [
            vlm_label(row["qwen_result"], include_authority=include_authority)
            for row in rows
        ],
        dtype=float,
    )
    glm = np.asarray(
        [
            vlm_label(row["glm_result"], include_authority=include_authority)
            for row in rows
        ],
        dtype=float,
    )
    text = np.asarray(
        [text_label(row["reaction"], row["context"], row["title"]) for row in rows],
        dtype=float,
    )
    clip_scan = np.asarray(
        [
            row.get("features", {}).get(
                "candidate_scan.active_nonnegative", 0
            )
            > 0
            for row in rows
        ],
        dtype=float,
    )
    scores = {
        "qwen_all_atoms": qwen,
        "glm_all_atoms": glm,
        "dual_vlm_intersection": (qwen + glm == 2).astype(float),
        "dual_vlm_union": (qwen + glm >= 1).astype(float),
        # At least two independent positive votes; a VLM must be one of them.
        "two_of_three_with_vlm": (
            ((qwen + glm + text) >= 2) & ((qwen + glm) >= 1)
        ).astype(float),
        # Proposal-only: expected to favor recall, never an acceptance rule.
        "clip_scan_candidate_generator": clip_scan,
    }
    report = {
        "items": len(rows),
        "positives": int(gold.sum()),
        "metrics": {
            name: metric_record(gold, score) for name, score in scores.items()
        },
    }
    errors = []
    selected = scores["two_of_three_with_vlm"]
    for row, truth, pred in zip(rows, gold, selected, strict=True):
        if bool(truth) != bool(pred):
            errors.append(
                {
                    "item_id": row["item_id"],
                    "gold": bool(truth),
                    "predicted": bool(pred),
                    "reaction": row["reaction"],
                    "manual_role": row["manual"].get("reaction_source_role"),
                    "manual_content": row["manual"].get("reaction_content"),
                    "manual_temporal": row["manual"].get("temporal_relation"),
                    "manual_description": row["manual"].get("description"),
                }
            )
    return report, errors


def evaluate_transfer_model(
    training_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
    *,
    include_authority: bool,
) -> dict[str, Any]:
    """Fit only on the earlier 58-item audit; score the untouched 24-item audit."""
    training_rows = [
        row
        for row in training_rows
        if row["features"].get("meta.reaction_strength", 0) > 0
    ]
    prefixes = (
        "text.",
        "qwen.",
        "glm.",
        "intervention.",
        "speaker.",
        "structure.",
        "roles.",
        "candidate_scan.",
        "acoustic.",
        "face_tracks.",
    )
    names = sorted(
        name
        for name in set.intersection(
            *[
                set(row["features"])
                for row in training_rows + holdout_rows
            ]
        )
        if name.startswith(prefixes)
    )
    x_train = np.asarray(
        [[row["features"][name] for name in names] for row in training_rows],
        dtype=float,
    )
    x_test = np.asarray(
        [[row["features"][name] for name in names] for row in holdout_rows],
        dtype=float,
    )
    y_train = np.asarray([row["gold"] for row in training_rows], dtype=int)
    y_test = np.asarray(
        [
            manual_target(row["manual"], include_authority=include_authority)
            for row in holdout_rows
        ],
        dtype=int,
    )
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=0.2, class_weight="balanced", max_iter=2000, random_state=29
        ),
    )
    model.fit(x_train, y_train)
    score = model.predict_proba(x_test)[:, 1]
    return {
        "training_items": len(training_rows),
        "features": names,
        "metrics": metric_record(y_test, score),
    }


def load_training_rows(root: Path) -> list[dict[str, Any]]:
    return build_training_rows(
        json.loads((root / "manifest.json").read_text()),
        json.loads((root / "manual_revealed_label_adjudication.json").read_text()),
        load_jsonl(root / "manual_blind_witnessed_review.jsonl"),
        load_jsonl(root / "qwen_v7c_full.jsonl"),
        load_jsonl(root / "glm_v7c_full.jsonl"),
        json.loads((root / "all_clip_transcripts_tiny_en.json").read_text()),
        root,
    )


def feature_availability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    names = sorted(set().union(*(row["features"] for row in rows)))
    return {
        name: {
            "observed": sum(
                math.isfinite(float(row["features"].get(name, math.nan)))
                for row in rows
            ),
            "nonzero": sum(
                math.isfinite(float(row["features"].get(name, math.nan)))
                and float(row["features"].get(name, 0)) != 0
                for row in rows
            ),
        }
        for name in names
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-root", type=Path, required=True)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--candidate-out",
        type=Path,
        help="optional append-free JSONL of candidate windows for the next VLM stage",
    )
    parser.add_argument("--skip-media", action="store_true")
    args = parser.parse_args()

    rows = build_holdout_rows(
        args.holdout_root, extract_media_features=not args.skip_media
    )
    strict, strict_errors = evaluate_frozen_rules(
        rows, include_authority=False
    )
    extended, extended_errors = evaluate_frozen_rules(
        rows, include_authority=True
    )
    report: dict[str, Any] = {
        "kind": "witnessed_reaction_source_disjoint_holdout_v1",
        "policy": "audit_only_no_corpus_mutation",
        "holdout_source_overlap_with_training": 0,
        "gold_policy": {
            "strict": "bystander/audience + grounded + targeted + ordered/overlap",
            "extended": "strict plus authority/host intervention",
        },
        "frozen_rules": {
            "strict_bystander": strict,
            "extended_third_party": extended,
        },
        "selected_rule_errors": {
            "strict_bystander": strict_errors,
            "extended_third_party": extended_errors,
        },
        "feature_availability": feature_availability(rows),
    }
    if args.training_root:
        training_rows = load_training_rows(args.training_root)
        report["trained_on_prior_audit_only"] = {
            "strict_bystander": evaluate_transfer_model(
                training_rows, rows, include_authority=False
            ),
            "extended_third_party": evaluate_transfer_model(
                training_rows, rows, include_authority=True
            ),
        }
    if args.candidate_out:
        args.candidate_out.parent.mkdir(parents=True, exist_ok=True)
        with args.candidate_out.open("w") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        {
                            "item_id": row["item_id"],
                            "uid": row["uid"],
                            "reaction": row["reaction"],
                            "candidates": row["candidates"],
                            "policy": "proposal_only_not_an_acceptance_label",
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
