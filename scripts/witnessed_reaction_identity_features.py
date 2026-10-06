#!/usr/bin/env python3
"""Identity-conditioned weak features for witnessed bystander reactions.

All functions are deterministic and shadow-only.  Explicit speaker IDs are
kept separate from acoustic speaker-change proxies.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import librosa
import numpy as np

if __package__:
    from scripts.witnessed_reaction_text_features import intervention_features
else:
    from witnessed_reaction_text_features import intervention_features


def speaker_turn_features(
    segments: list[dict[str, Any]],
    reaction_index: int | None,
    *,
    action_speaker: str | None = None,
) -> dict[str, float]:
    """Use real diarization labels when present; otherwise abstain."""
    out = {
        "speaker.explicit_available": 0.0,
        "speaker.changed_at_reaction": math.nan,
        "speaker.reactor_differs_from_action": math.nan,
        "speaker.prior_distinct_speakers": math.nan,
        "speaker.reaction_overlap": math.nan,
    }
    if reaction_index is None or not (0 <= reaction_index < len(segments)):
        return out
    reactor = segments[reaction_index].get("speaker")
    labeled = [s for s in segments if s.get("speaker") is not None]
    if reactor is None or not labeled:
        return out
    previous = next(
        (
            segments[i].get("speaker")
            for i in range(reaction_index - 1, -1, -1)
            if segments[i].get("speaker") is not None
        ),
        None,
    )
    prior_speakers = {
        s.get("speaker")
        for s in segments[:reaction_index]
        if s.get("speaker") is not None
    }
    start = float(segments[reaction_index].get("start") or 0)
    overlap = any(
        float(s.get("end") or 0) > start
        for s in segments[:reaction_index]
        if s.get("speaker") != reactor
    )
    out.update(
        {
            "speaker.explicit_available": 1.0,
            "speaker.changed_at_reaction": float(
                previous is not None and previous != reactor
            ),
            "speaker.reactor_differs_from_action": (
                float(reactor != action_speaker)
                if action_speaker is not None
                else math.nan
            ),
            "speaker.prior_distinct_speakers": float(len(prior_speakers)),
            "speaker.reaction_overlap": float(overlap),
        }
    )
    return out


def temporal_structure_features(
    segments: list[dict[str, Any]], reaction_index: int | None
) -> dict[str, float]:
    out = {
        "structure.reaction_localized": 0.0,
        "structure.response_latency": math.nan,
        "structure.reaction_duration": math.nan,
        "structure.preceding_turns_3s": math.nan,
        "structure.short_response": math.nan,
    }
    if reaction_index is None or not (0 <= reaction_index < len(segments)):
        return out
    current = segments[reaction_index]
    start = float(current.get("start") or 0)
    end = float(current.get("end") or start)
    prior_end = (
        float(segments[reaction_index - 1].get("end") or start)
        if reaction_index
        else 0.0
    )
    words = re.findall(r"\w+", str(current.get("text") or ""))
    out.update(
        {
            "structure.reaction_localized": 1.0,
            "structure.response_latency": max(-2.0, min(10.0, start - prior_end)),
            "structure.reaction_duration": max(0.0, end - start),
            "structure.preceding_turns_3s": float(
                sum(
                    float(s.get("end") or 0) >= start - 3
                    for s in segments[:reaction_index]
                )
            ),
            "structure.short_response": float(0 < len(words) <= 8),
        }
    )
    return out


def role_separation_features(
    violator_role: str | None,
    reactor_role: str | None,
    *,
    vlm_roles: list[str | None] = (),
) -> dict[str, float]:
    """Represent role separation without deciding the final label."""
    victim_roles = {"victim", "affected_target"}
    third_party_roles = {
        "bystander",
        "third_party",
        "organic_audience",
        "authority_or_host",
    }
    normalized_reactor = str(reactor_role or "").lower()
    normalized_violator = str(violator_role or "").lower()
    model_third = [role in third_party_roles for role in vlm_roles if role]
    return {
        "roles.metadata_available": float(bool(reactor_role or violator_role)),
        "roles.reactor_third_party": float(normalized_reactor in third_party_roles),
        "roles.reactor_affected_target": float(normalized_reactor in victim_roles),
        "roles.reactor_is_camera": float(normalized_reactor == "camera_person"),
        "roles.roles_differ": float(
            bool(normalized_reactor)
            and bool(normalized_violator)
            and normalized_reactor != normalized_violator
        ),
        "roles.vlm_third_party_votes": float(sum(model_third)),
        "roles.vlm_third_party_agreement": (
            float(len(model_third) >= 2 and all(model_third))
        ),
    }


def _cosine_distance(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 1.0 - float(np.dot(left, right) / denom) if denom else math.nan


def acoustic_identity_features(
    clip: Path, boundary: float | None
) -> dict[str, float]:
    """Acoustic speaker-change and vocal-baseline proxies (not diarization)."""
    empty = {
        "acoustic.proxy_available": 0.0,
        "acoustic.mfcc_change": math.nan,
        "acoustic.pitch_change": math.nan,
        "acoustic.vocal_energy_z": math.nan,
        "acoustic.spectral_change": math.nan,
    }
    if boundary is None or not clip.exists() or boundary < 1.25:
        return empty
    try:
        y, sr = librosa.load(
            str(clip), sr=16000, mono=True,
            offset=max(0.0, boundary - 2.5), duration=5.0
        )
    except Exception:
        return empty
    center = min(int(2.5 * sr), len(y))
    pre, post = y[max(0, center - 2 * sr) : center], y[center : center + 2 * sr]
    if min(len(pre), len(post)) < sr // 2:
        return empty
    pre_mfcc = np.mean(librosa.feature.mfcc(y=pre, sr=sr, n_mfcc=13), axis=1)
    post_mfcc = np.mean(librosa.feature.mfcc(y=post, sr=sr, n_mfcc=13), axis=1)
    pre_cent = float(np.mean(librosa.feature.spectral_centroid(y=pre, sr=sr)))
    post_cent = float(np.mean(librosa.feature.spectral_centroid(y=post, sr=sr)))
    pre_rms = librosa.feature.rms(y=pre)[0]
    post_rms = librosa.feature.rms(y=post)[0]

    def median_pitch(signal: np.ndarray) -> float:
        pitch = librosa.yin(signal, fmin=70, fmax=500, sr=sr)
        finite = pitch[np.isfinite(pitch)]
        return float(np.median(finite)) if len(finite) else math.nan

    pre_pitch, post_pitch = median_pitch(pre), median_pitch(post)
    return {
        "acoustic.proxy_available": 1.0,
        "acoustic.mfcc_change": _cosine_distance(pre_mfcc, post_mfcc),
        "acoustic.pitch_change": (
            abs(math.log2(post_pitch / pre_pitch))
            if pre_pitch > 0 and post_pitch > 0
            else math.nan
        ),
        "acoustic.vocal_energy_z": float(
            (np.mean(post_rms) - np.mean(pre_rms))
            / (np.std(pre_rms) + 1e-6)
        ),
        "acoustic.spectral_change": abs(post_cent - pre_cent) / 4000.0,
    }


@dataclass
class FaceObservation:
    time: float
    box: tuple[float, float, float, float]
    crop: np.ndarray
    gaze_proxy: tuple[float, float]


@dataclass
class FaceTrack:
    observations: list[FaceObservation] = field(default_factory=list)


def box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = lw * lh + rw * rh - intersection
    return intersection / union if union > 0 else 0.0


def gaze_proxy(crop: np.ndarray) -> tuple[float, float]:
    """Return upper-face darkness centroid as a weak head/gaze proxy."""
    gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    upper = gray[: max(1, gray.shape[0] // 2)].astype(float)
    weights = 255.0 - upper
    total = float(weights.sum())
    if total <= 0:
        return (0.5, 0.5)
    ys, xs = np.indices(weights.shape)
    return (
        float((xs * weights).sum() / total / max(1, weights.shape[1] - 1)),
        float((ys * weights).sum() / total / max(1, weights.shape[0] - 1)),
    )


def link_face_observations(
    frames: list[list[FaceObservation]], *, minimum_iou: float = 0.15
) -> list[FaceTrack]:
    tracks: list[FaceTrack] = []
    for observations in frames:
        unmatched = set(range(len(observations)))
        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(tracks):
            if not track.observations:
                continue
            for obs_index, observation in enumerate(observations):
                candidates.append(
                    (
                        box_iou(track.observations[-1].box, observation.box),
                        track_index,
                        obs_index,
                    )
                )
        used_tracks: set[int] = set()
        for iou, track_index, obs_index in sorted(candidates, reverse=True):
            if (
                iou < minimum_iou
                or track_index in used_tracks
                or obs_index not in unmatched
            ):
                continue
            tracks[track_index].observations.append(observations[obs_index])
            used_tracks.add(track_index)
            unmatched.remove(obs_index)
        tracks.extend(FaceTrack([observations[i]]) for i in sorted(unmatched))
    return tracks


def summarize_face_tracks(
    tracks: list[FaceTrack], boundary: float
) -> dict[str, float]:
    empty = {
        "face_tracks.available": 0.0,
        "face_tracks.cross_boundary_tracks": 0.0,
        "face_tracks.max_head_motion_delta": math.nan,
        "face_tracks.max_gaze_change": math.nan,
        "face_tracks.max_expression_change": math.nan,
        "face_tracks.synchronous_reactors": 0.0,
        "face_tracks.synchrony_fraction": math.nan,
    }
    deltas: list[tuple[float, float, float]] = []
    for track in tracks:
        pre = [o for o in track.observations if o.time < boundary]
        post = [o for o in track.observations if o.time >= boundary]
        if len(pre) < 2 or len(post) < 2:
            continue
        pre_center = np.asarray(
            [[o.box[0] + o.box[2] / 2, o.box[1] + o.box[3] / 2] for o in pre]
        )
        post_center = np.asarray(
            [[o.box[0] + o.box[2] / 2, o.box[1] + o.box[3] / 2] for o in post]
        )
        head_delta = float(
            np.linalg.norm(np.mean(post_center, axis=0) - np.mean(pre_center, axis=0))
        )
        pre_gaze = np.mean([o.gaze_proxy for o in pre], axis=0)
        post_gaze = np.mean([o.gaze_proxy for o in post], axis=0)
        gaze_delta = float(np.linalg.norm(post_gaze - pre_gaze))
        pre_crop = np.mean([o.crop.astype(float) for o in pre], axis=0)
        post_crop = np.mean([o.crop.astype(float) for o in post], axis=0)
        expression_delta = float(np.mean(np.abs(post_crop - pre_crop)) / 255.0)
        deltas.append((head_delta, gaze_delta, expression_delta))
    if not deltas:
        return empty
    array = np.asarray(deltas)
    # Conservative, resolution-normalized reaction proxy: either a meaningful
    # gaze/texture change or substantial track-center movement.
    reactive = (
        (array[:, 0] >= 8.0)
        | (array[:, 1] >= 0.08)
        | (array[:, 2] >= 0.10)
    )
    return {
        "face_tracks.available": 1.0,
        "face_tracks.cross_boundary_tracks": float(len(deltas)),
        "face_tracks.max_head_motion_delta": float(np.max(array[:, 0])),
        "face_tracks.max_gaze_change": float(np.max(array[:, 1])),
        "face_tracks.max_expression_change": float(np.max(array[:, 2])),
        "face_tracks.synchronous_reactors": float(np.sum(reactive)),
        "face_tracks.synchrony_fraction": float(np.mean(reactive)),
    }


def face_identity_features(
    clip: Path, boundary: float | None
) -> dict[str, float]:
    empty = summarize_face_tracks([], 0.0)
    if boundary is None or not clip.exists() or boundary < 1.5:
        return empty
    cap = cv2.VideoCapture(str(clip))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    count = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = count / fps if fps > 0 else 0
    if duration <= boundary:
        cap.release()
        return empty
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    frames: list[list[FaceObservation]] = []
    for sec in np.arange(max(0, boundary - 2.0), min(duration, boundary + 2.0), 0.25):
        cap.set(cv2.CAP_PROP_POS_MSEC, float(sec * 1000))
        ok, frame = cap.read()
        if not ok:
            frames.append([])
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes = cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=4, minSize=(24, 24)
        )
        observations = []
        for x, y, w, h in boxes:
            crop = cv2.resize(gray[y : y + h, x : x + w], (48, 48))
            observations.append(
                FaceObservation(
                    float(sec),
                    (float(x), float(y), float(w), float(h)),
                    crop,
                    gaze_proxy(crop),
                )
            )
        frames.append(observations)
    cap.release()
    return summarize_face_tracks(link_face_observations(frames), boundary)
