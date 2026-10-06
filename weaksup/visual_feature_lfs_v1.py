#!/usr/bin/env python3
"""Visual-evidence labeling functions from the corpus-wide cheap feature ledgers.

Consumes the append-only 20260724 full-corpus shadow features (low-level
motion/face stats for all pillars; pose person counts and CLIP scene prompts
for witnessed and instructional only — the recorded backfill policy excludes
pose/CLIP for commentary after a failed holdout, and this module honors that
exclusion).  Emits deterministic threshold votes in the ``visual_depiction`` /
``source_format_provenance`` families; the label model learns each family's
actual reliability, so these are weak evidence, never acceptance rules.

Key logical constraint now expressible: a scene with at most one visible
person cannot contain an independent bystander, while three or more visible
people make one physically possible.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from weaksup.labeling_functions_v1 import make_lf_record
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from weaksup.labeling_functions_v1 import make_lf_record


VISUAL_LF_VERSION = "visual_feature_lfs_v1"

# CLIP prompt indices in clip_scores (order frozen by the 20260724 backfill).
PROMPT_STAGED_ROLEPLAY = 0
PROMPT_REAL_INTERACTION = 1
PROMPT_TALKING_HEAD = 2
PROMPT_NEWS_LECTURE = 3
PROMPT_SLIDE_GRAPHIC = 4
PROMPT_GENERIC_BROLL = 5

# Thresholds are deliberately plain; the EM model weighs the family.
MULTI_PERSON_MEAN = 3.0
SOLO_PERSON_MEAN = 1.2
LOW_PRESENCE_FRACTION = 0.3
CLIP_SUPPORT_MIN = 0.45
CLIP_PRESENTATION_MIN = 0.5

_CLIP_FILE_INDEX = re.compile(r"(?:clip|demo)_(\d+)\.mp4$")


def _record(*, item_id, pillar, target, lf_id, family, vote, reason, evidence):
    return make_lf_record(
        item_id=item_id, pillar=pillar, target=target, lf_id=lf_id,
        family=family, vote=vote,
        abstain_reason=reason if vote == 0 else None,
        evidence=evidence, model_or_rule_version=VISUAL_LF_VERSION,
    )


def load_feature_index(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    """Index feature rows by (pillar, uid, clip/demo index).

    The index is parsed from the media filename (clip_N / demo_N), not the
    ledger's own item counter, so alignment with pipeline items is exact.
    Commentary rows (whole-source media) index as 0.
    """
    index: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not path.is_file():
        return index
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error"):
                continue
            uid, pillar = row.get("uid"), row.get("pillar")
            if not uid or not pillar:
                continue
            match = _CLIP_FILE_INDEX.search(str(row.get("clip") or ""))
            idx = int(match.group(1)) if match else 0
            index[(pillar, uid, idx)] = {
                "low_level": row.get("low_level") or {},
                "keypoints": row.get("keypoints") or {},
                "clip_scores": row.get("clip_scores") or {},
            }
    return index


def _clip_means(features: dict[str, Any]) -> list[float] | None:
    means = (features.get("clip_scores") or {}).get("mean_probabilities")
    if isinstance(means, list) and len(means) >= 6:
        return [float(value) for value in means]
    return None


def witnessed_visual_lf_records(
    item_id: str, features: dict[str, Any]
) -> list[dict[str, Any]]:
    records = []
    keypoints = features.get("keypoints") or {}
    count = keypoints.get("person_count_mean")
    presence = keypoints.get("person_present_fraction")
    vote, reason = 0, "person_counts_unavailable_or_ambiguous"
    if isinstance(count, (int, float)):
        if count >= MULTI_PERSON_MEAN:
            vote = 1
        elif count <= SOLO_PERSON_MEAN or (
            isinstance(presence, (int, float)) and presence < LOW_PRESENCE_FRACTION
        ):
            vote = -1  # nobody besides the actor is visible: no third party
    records.append(
        _record(
            item_id=item_id, pillar="witnessed",
            target="independent_bystander_signal",
            lf_id="vis_witnessed_person_count_v1", family="visual_depiction",
            vote=vote, reason=reason,
            evidence={"person_count_mean": count, "person_present_fraction": presence},
        )
    )
    means = _clip_means(features)
    if means is None:
        interaction_vote, staged_vote = 0, 0
        interaction_evidence: dict[str, Any] = {"clip_scores": None}
    else:
        interaction = means[PROMPT_REAL_INTERACTION]
        presentation = max(
            means[PROMPT_TALKING_HEAD], means[PROMPT_NEWS_LECTURE],
            means[PROMPT_SLIDE_GRAPHIC], means[PROMPT_GENERIC_BROLL],
        )
        staged = means[PROMPT_STAGED_ROLEPLAY]
        interaction_vote = (
            1 if interaction >= CLIP_SUPPORT_MIN and interaction > presentation
            else -1 if presentation >= CLIP_PRESENTATION_MIN and presentation > interaction
            else 0
        )
        staged_vote = -1 if staged >= CLIP_PRESENTATION_MIN and staged > interaction else 0
        interaction_evidence = {
            "interaction": interaction, "presentation": presentation, "staged": staged,
        }
    records.append(
        _record(
            item_id=item_id, pillar="witnessed", target="norm_event_supported",
            lf_id="vis_witnessed_interaction_scene_v1", family="visual_depiction",
            vote=interaction_vote, reason="clip_scores_unavailable_or_ambiguous",
            evidence=interaction_evidence,
        )
    )
    records.append(
        _record(
            item_id=item_id, pillar="witnessed",
            target="independent_bystander_signal",
            lf_id="vis_witnessed_staged_roleplay_v1",
            family="source_format_provenance",
            vote=staged_vote, reason="no_dominant_staged_roleplay_score",
            evidence=interaction_evidence,
        )
    )
    return records


def instructional_visual_lf_records(
    item_id: str, features: dict[str, Any]
) -> list[dict[str, Any]]:
    """Any performed format counts as a demo, so staged role-play is POSITIVE
    support here (unlike the witnessed organic subtype)."""
    records = []
    means = _clip_means(features)
    if means is None:
        vote, evidence = 0, {"clip_scores": None}
    else:
        demo_support = max(
            means[PROMPT_STAGED_ROLEPLAY], means[PROMPT_REAL_INTERACTION]
        )
        presentation = max(
            means[PROMPT_TALKING_HEAD], means[PROMPT_NEWS_LECTURE],
            means[PROMPT_SLIDE_GRAPHIC],
        )
        broll = means[PROMPT_GENERIC_BROLL]
        vote = (
            1 if demo_support >= CLIP_SUPPORT_MIN and demo_support > max(presentation, broll)
            else -1 if max(presentation, broll) >= CLIP_PRESENTATION_MIN
            and max(presentation, broll) > demo_support
            else 0
        )
        evidence = {
            "demo_support": demo_support, "presentation": presentation, "broll": broll,
        }
    records.append(
        _record(
            item_id=item_id, pillar="instructional", target="demonstration_present",
            lf_id="vis_instructional_demo_scene_v1", family="visual_depiction",
            vote=vote, reason="clip_scores_unavailable_or_ambiguous",
            evidence=evidence,
        )
    )
    keypoints = features.get("keypoints") or {}
    pair = keypoints.get("multiple_people_fraction")
    presence = keypoints.get("person_present_fraction")
    vote = 0
    if isinstance(pair, (int, float)) and pair >= 0.5:
        vote = 1
    elif isinstance(presence, (int, float)) and presence < 0.2:
        vote = -1
    records.append(
        _record(
            item_id=item_id, pillar="instructional", target="demonstration_present",
            lf_id="vis_instructional_person_pair_v1", family="visual_depiction",
            vote=vote, reason="person_presence_ambiguous",
            evidence={"multiple_people_fraction": pair, "person_present_fraction": presence},
        )
    )
    return records


def commentary_visual_lf_records(
    item_id: str, features: dict[str, Any]
) -> list[dict[str, Any]]:
    """Low-level only (pose/CLIP are policy-excluded for commentary).  Votes
    on ``event_present_in_source``: a moving multi-face scene may contain the
    discussed event; a static faceless frame is graphics/podcast art."""
    low = features.get("low_level") or {}
    faces = low.get("face_present_fraction")
    multi = low.get("multiple_faces_fraction")
    motion = low.get("motion_mean")
    vote = 0
    if isinstance(faces, (int, float)) and isinstance(multi, (int, float)):
        if faces >= 0.5 and multi >= 0.2:
            vote = 1
        elif faces < 0.1 and isinstance(motion, (int, float)) and motion < 0.05:
            vote = -1
    return [
        _record(
            item_id=item_id, pillar="commentary", target="event_present_in_source",
            lf_id="vis_commentary_scene_activity_v1", family="visual_depiction",
            vote=vote, reason="face_motion_evidence_ambiguous",
            evidence={
                "face_present_fraction": faces,
                "multiple_faces_fraction": multi,
                "motion_mean": motion,
            },
        )
    ]
