#!/usr/bin/env python3
"""Benchmark reaction-focused weak signals on the witnessed manual audit.

The target is deliberately narrower than the witnessed clip acceptance
contract: a distinct third party gives a contemporaneous, targeted response to
somebody else's behavior.  Authenticity, tacit-norm validity, clean splicing,
and label correctness are downstream tasks and are not part of this target.

This script is audit-only.  It does not touch the corpus.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

if __package__:
    from scripts.witnessed_reaction_candidate_scan import (
        candidate_scan_features,
        scan_reaction_candidates,
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
    from witnessed_reaction_candidate_scan import (
        candidate_scan_features,
        scan_reaction_candidates,
    )
    from witnessed_reaction_identity_features import (
        acoustic_identity_features,
        face_identity_features,
        intervention_features,
        role_separation_features,
        speaker_turn_features,
        temporal_structure_features,
    )

TOKEN = re.compile(r"[a-z']+")
DIRECT_OBJECTION = re.compile(
    r"\b(?:stop|don'?t|do not|no\b|enough|shut (?:up|your mouth)|"
    r"leave (?:him|her|them|me) alone|get (?:away|back|off|out)|"
    r"calm down|cut it out|knock it off|you can'?t|you shouldn'?t|"
    r"that'?s not (?:okay|right)|what are you doing)\b",
    re.I,
)
PROTECTIVE = re.compile(
    r"\b(?:are you (?:okay|alright)|help (?:him|her|them)|"
    r"call (?:the )?(?:police|security)|i'?ll report|back off)\b",
    re.I,
)
DIRECT_ADDRESS = re.compile(
    r"\b(?:you|your|sir|ma'?am|bro|dude|man|buddy|hey)\b", re.I
)
GENERIC_AFFECT = re.compile(
    r"^(?:\W*(?:oh|wow|whoa|woah|gosh|god|ah|ha|haha|lol|"
    r"what|no|yes|yeah|okay|omg)\W*){1,5}$",
    re.I,
)
REPORTED = re.compile(
    r"\b(?:he said|she said|they said|according to|reportedly|"
    r"the video shows|you can see|we see|this footage|police say)\b",
    re.I,
)
THIRD_PARTY_ROLES = {"bystander", "organic_audience", "authority_or_host"}
TARGETED_CONTENT = {
    "targeted_objection",
    "correction_or_sanction",
    "protective_intervention",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def token_set(text: str) -> set[str]:
    return set(TOKEN.findall(text.lower()))


def overlap(a: str, b: str) -> float:
    left, right = token_set(a), token_set(b)
    if not left or not right:
        return 0.0
    return 2 * len(left & right) / (len(left) + len(right))


def reaction_segment(
    reaction: str, context: str, segments: list[dict[str, Any]]
) -> tuple[dict[str, Any] | None, float]:
    best: tuple[dict[str, Any] | None, float] = (None, 0.0)
    for segment in segments:
        text = str(segment.get("text") or "")
        score = max(overlap(reaction, text), 0.65 * overlap(context, text))
        if score > best[1]:
            best = (segment, score)
    return best


def audio_features(
    clip: Path, boundary: float | None
) -> dict[str, float]:
    if boundary is None or not clip.exists():
        return {
            "audio.boundary_available": 0.0,
            "audio.rms_delta_db": math.nan,
            "audio.onset_delta": math.nan,
            "audio.post_rms_cv": math.nan,
        }
    y, sr = librosa.load(str(clip), sr=16000, mono=True)
    center = min(max(int(boundary * sr), 0), len(y))
    span = 2 * sr
    pre = y[max(0, center - span) : center]
    post = y[center : min(len(y), center + span)]
    if len(pre) < sr // 4 or len(post) < sr // 4:
        return {
            "audio.boundary_available": 0.0,
            "audio.rms_delta_db": math.nan,
            "audio.onset_delta": math.nan,
            "audio.post_rms_cv": math.nan,
        }

    def rms(x: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(x)) + 1e-12))

    pre_rms, post_rms = rms(pre), rms(post)
    pre_onset = librosa.onset.onset_strength(y=pre, sr=sr)
    post_onset = librosa.onset.onset_strength(y=post, sr=sr)
    post_frames = librosa.feature.rms(y=post)[0]
    return {
        "audio.boundary_available": 1.0,
        "audio.rms_delta_db": float(
            20 * np.log10((post_rms + 1e-7) / (pre_rms + 1e-7))
        ),
        "audio.onset_delta": float(
            np.mean(post_onset) - np.mean(pre_onset)
        ),
        "audio.post_rms_cv": float(
            np.std(post_frames) / (np.mean(post_frames) + 1e-7)
        ),
    }


def visual_temporal_features(
    clip: Path, boundary: float | None
) -> dict[str, float]:
    empty = {
        "temporal.boundary_available": 0.0,
        "temporal.motion_delta": math.nan,
        "temporal.onset_motion_peak": math.nan,
        "temporal.cut_at_onset": math.nan,
        "temporal.face_count_delta": math.nan,
        "temporal.face_motion_delta": math.nan,
    }
    if boundary is None or not clip.exists():
        return empty
    cap = cv2.VideoCapture(str(clip))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else 0
    if duration <= 0 or boundary < 1.0 or boundary >= duration - 0.5:
        cap.release()
        return empty
    times = np.linspace(max(0, boundary - 2.0), min(duration, boundary + 2.0), 17)
    frames: list[np.ndarray] = []
    face_counts: list[int] = []
    face_regions: list[np.ndarray | None] = []
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    for sec in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, float(sec * 1000))
        ok, frame = cap.read()
        if not ok:
            continue
        gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray_full, (160, 90))
        frames.append(gray)
        faces = cascade.detectMultiScale(
            gray_full, scaleFactor=1.2, minNeighbors=4, minSize=(24, 24)
        )
        face_counts.append(len(faces))
        if len(faces):
            x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
            crop = gray_full[y : y + h, x : x + w]
            face_regions.append(cv2.resize(crop, (48, 48)))
        else:
            face_regions.append(None)
    cap.release()
    if len(frames) < 8:
        return empty
    diffs = np.asarray(
        [
            float(np.mean(cv2.absdiff(frames[i - 1], frames[i]))) / 255.0
            for i in range(1, len(frames))
        ]
    )
    hist_diffs = []
    for i in range(1, len(frames)):
        left = cv2.calcHist([frames[i - 1]], [0], None, [32], [0, 256])
        right = cv2.calcHist([frames[i]], [0], None, [32], [0, 256])
        cv2.normalize(left, left)
        cv2.normalize(right, right)
        hist_diffs.append(
            float(cv2.compareHist(left, right, cv2.HISTCMP_BHATTACHARYYA))
        )
    mid = len(diffs) // 2
    pre = diffs[:mid] if mid else diffs
    post = diffs[mid:] if mid < len(diffs) else diffs
    face_motion = np.asarray(
        [
            (
                float(np.mean(cv2.absdiff(face_regions[i - 1], face_regions[i])))
                / 255.0
                if face_regions[i - 1] is not None and face_regions[i] is not None
                else math.nan
            )
            for i in range(1, len(face_regions))
        ]
    )
    pre_face = face_motion[:mid]
    post_face = face_motion[mid:]
    return {
        "temporal.boundary_available": 1.0,
        "temporal.motion_delta": float(np.mean(post) - np.mean(pre)),
        "temporal.onset_motion_peak": float(
            np.max(diffs[max(0, mid - 1) : min(len(diffs), mid + 2)])
        ),
        "temporal.cut_at_onset": float(
            np.max(hist_diffs[max(0, mid - 1) : min(len(hist_diffs), mid + 2)])
        ),
        "temporal.face_count_delta": float(
            np.mean(face_counts[len(face_counts) // 2 :])
            - np.mean(face_counts[: len(face_counts) // 2])
        ),
        "temporal.face_motion_delta": float(
            np.nanmean(post_face) - np.nanmean(pre_face)
        )
        if np.isfinite(pre_face).any() and np.isfinite(post_face).any()
        else math.nan,
    }


def vlm_features(prefix: str, row: dict[str, Any] | None) -> dict[str, float]:
    result = (row or {}).get("result") or {}
    return {
        f"{prefix}.grounded": float(
            result.get("reaction_visible_or_audibly_grounded") == "yes"
        ),
        f"{prefix}.third_party": float(
            result.get("reaction_source_role") in THIRD_PARTY_ROLES
        ),
        f"{prefix}.targeted_content": float(
            result.get("reaction_content") in TARGETED_CONTENT
        ),
        f"{prefix}.targets_action": float(
            result.get("reaction_targets_action") == "yes"
        ),
        f"{prefix}.ordered": float(
            result.get("action_established_before_reaction") == "yes"
        ),
    }


def build_rows(
    manifest: dict[str, Any],
    adjudication: dict[str, Any],
    blind: list[dict[str, Any]],
    qwen: list[dict[str, Any]],
    glm: list[dict[str, Any]],
    transcripts: dict[str, Any],
    audit_root: Path,
) -> list[dict[str, Any]]:
    failed = set(
        adjudication["failure_groups"][
            "detected_reaction_is_not_targeted_third_party_disapproval_or_intervention"
        ]
    )
    qwen_by = {row["item_id"]: row for row in qwen}
    glm_by = {row["item_id"]: row for row in glm}
    blind_by = {row["audit_index"]: row for row in blind}
    transcript_by = {
        str(row["uid"]).split("_", 1)[-1]: row.get("segments") or []
        for row in transcripts["records"]
    }
    rows = []
    for item in manifest["visual_samples"]:
        idx, uid = int(item["audit_index"]), str(item["uid"])
        item_id = f"witnessed:{uid}:0"
        reaction = str(item.get("reaction") or "")
        context = str(item.get("reaction_context") or "")
        title = str(item.get("title") or "")
        segments = transcript_by.get(uid, [])
        segment, alignment = reaction_segment(reaction, context, segments)
        reaction_index = (
            segments.index(segment) if segment is not None else None
        )
        seg_text = str((segment or {}).get("text") or "")
        seg_start = (
            float(segment["start"])
            if segment is not None and segment.get("start") is not None
            else None
        )
        transcript_text = " ".join(str(s.get("text") or "") for s in segments)
        text = f"{reaction} {context}"
        words = TOKEN.findall(reaction.lower())
        features = {
            "meta.reaction_strength": float(item.get("reaction_strength") or 0),
            "meta.people": float(item.get("n_people") or 0),
            "meta.role_bystander": float(
                str(item.get("reactor_role") or "").lower()
                in {"bystander", "witness", "crowd", "audience"}
            ),
            "text.direct_objection": float(bool(DIRECT_OBJECTION.search(text))),
            "text.protective": float(bool(PROTECTIVE.search(text))),
            "text.direct_address": float(bool(DIRECT_ADDRESS.search(text))),
            "text.generic_affect_only": float(bool(GENERIC_AFFECT.match(reaction))),
            "text.reported": float(bool(REPORTED.search(f"{title} {transcript_text}"))),
            "text.reaction_words": float(len(words)),
            "text.negation_count": float(
                sum(word in {"no", "not", "don't", "stop", "never"} for word in words)
            ),
            "text.alignment": alignment,
            "text.segment_short": float(0 < len(TOKEN.findall(seg_text)) <= 8),
            "text.has_prior_context": float(
                seg_start is not None and seg_start >= 1.0
            ),
        }
        features.update(vlm_features("qwen", qwen_by.get(item_id)))
        features.update(vlm_features("glm", glm_by.get(item_id)))
        qwen_result = (qwen_by.get(item_id) or {}).get("result") or {}
        glm_result = (glm_by.get(item_id) or {}).get("result") or {}
        action_speaker = (
            segments[reaction_index - 1].get("speaker")
            if reaction_index is not None and reaction_index > 0
            else None
        )
        features.update(intervention_features(reaction))
        features.update(
            speaker_turn_features(
                segments, reaction_index, action_speaker=action_speaker
            )
        )
        features.update(temporal_structure_features(segments, reaction_index))
        features.update(
            candidate_scan_features(
                scan_reaction_candidates(
                    segments,
                    selected_quote=reaction,
                ),
                selected_boundary=seg_start,
            )
        )
        features.update(
            role_separation_features(
                item.get("violator_role"),
                item.get("reactor_role"),
                vlm_roles=[
                    qwen_result.get("reaction_source_role"),
                    glm_result.get("reaction_source_role"),
                ],
            )
        )
        clip = audit_root / "clips" / f"{idx}_{uid}.mp4"
        features.update(audio_features(clip, seg_start))
        features.update(visual_temporal_features(clip, seg_start))
        features.update(acoustic_identity_features(clip, seg_start))
        features.update(face_identity_features(clip, seg_start))
        cut = features.get("temporal.cut_at_onset", math.nan)
        features["structure.scene_cut_abstain"] = float(
            np.isfinite(cut) and cut >= 0.55
        )
        rows.append(
            {
                "audit_index": idx,
                "uid": uid,
                "item_id": item_id,
                "gold": idx not in failed,
                "reaction": reaction,
                "description": blind_by.get(idx, {}).get("description", ""),
                "features": features,
            }
        )
    return rows


def scalar_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    pred = score >= 0.5
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, pred, average="binary", zero_division=0
    )
    return {
        "average_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)),
        "precision_at_0_5": float(precision),
        "recall_at_0_5": float(recall),
        "f1_at_0_5": float(f1),
        "predicted_positive": int(pred.sum()),
    }


def feature_matrix(
    rows: list[dict[str, Any]], prefixes: tuple[str, ...]
) -> tuple[list[str], np.ndarray]:
    names = sorted(
        name
        for name in rows[0]["features"]
        if name.startswith(prefixes)
    )
    return names, np.asarray(
        [[row["features"][name] for name in names] for row in rows],
        dtype=float,
    )


def evaluate_models(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y = np.asarray([row["gold"] for row in rows], dtype=int)
    groups = np.asarray([row["uid"] for row in rows])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=29)
    feature_groups = {
        "metadata_text": ("meta.", "text."),
        "vlm_atomic": ("qwen.", "glm."),
        "audio": ("audio.",),
        "reaction_temporal": ("audio.", "temporal."),
        "intervention_structure": (
            "intervention.",
            "speaker.",
            "structure.",
            "roles.",
            "candidate_scan.",
        ),
        "identity_proxies": ("acoustic.", "face_tracks."),
        "all": (
            "meta.",
            "text.",
            "qwen.",
            "glm.",
            "audio.",
            "temporal.",
            "intervention.",
            "speaker.",
            "structure.",
            "roles.",
            "candidate_scan.",
            "acoustic.",
            "face_tracks.",
        ),
    }
    reports: dict[str, Any] = {}
    all_scores: dict[str, np.ndarray] = {}
    for group_name, prefixes in feature_groups.items():
        names, x = feature_matrix(rows, prefixes)
        estimators = {
            "logistic": make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(C=0.2, class_weight="balanced", max_iter=2000),
            ),
            "random_forest": make_pipeline(
                SimpleImputer(strategy="median"),
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=3,
                    min_samples_leaf=4,
                    class_weight="balanced",
                    random_state=29,
                ),
            ),
        }
        reports[group_name] = {"features": names, "models": {}}
        for model_name, estimator in estimators.items():
            score = cross_val_predict(
                estimator,
                x,
                y,
                groups=groups,
                cv=cv,
                method="predict_proba",
            )[:, 1]
            reports[group_name]["models"][model_name] = scalar_metrics(y, score)
            all_scores[f"{group_name}.{model_name}"] = score

    best_name = max(
        all_scores,
        key=lambda name: average_precision_score(y, all_scores[name]),
    )
    best_score = all_scores[best_name]
    errors = []
    for row, score in zip(rows, best_score, strict=True):
        if (score >= 0.5) != bool(row["gold"]):
            errors.append(
                {
                    "audit_index": row["audit_index"],
                    "uid": row["uid"],
                    "gold": row["gold"],
                    "score": float(score),
                    "reaction": row["reaction"],
                    "description": row["description"],
                }
            )
    return {
        "target": (
            "distinct third party gives a contemporaneous targeted response "
            "to another person's behavior"
        ),
        "items": len(rows),
        "positives": int(y.sum()),
        "evaluation": "five_fold_source_grouped_out_of_fold",
        "feature_groups": reports,
        "best_oof_model": best_name,
        "best_oof_errors_at_0_5": errors,
        "policy": "shadow_only_no_corpus_mutation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    rows = build_rows(
        json.loads((root / "manifest.json").read_text()),
        json.loads((root / "manual_revealed_label_adjudication.json").read_text()),
        load_jsonl(root / "manual_blind_witnessed_review.jsonl"),
        load_jsonl(root / "qwen_v7c_full.jsonl"),
        load_jsonl(root / "glm_v7c_full.jsonl"),
        json.loads((root / "all_clip_transcripts_tiny_en.json").read_text()),
        root,
    )
    complete = [
        row for row in rows
        if row["features"]["meta.reaction_strength"] > 0
    ]
    report = {
        "kind": "witnessed_reaction_feature_cycles_v2",
        "gold_derivation": (
            "manual adjudication complement of "
            "detected_reaction_is_not_targeted_third_party_disapproval_or_intervention"
        ),
        "warning": (
            "The full cohort has a metadata-missingness confound; use the "
            "metadata-complete cohort for conclusions."
        ),
        "cohorts": {
            "all_with_missingness_confound": evaluate_models(rows),
            "metadata_complete": evaluate_models(complete),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
