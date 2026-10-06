#!/usr/bin/env python3
"""Materialize standardized LF records from the real corpus metadata.

Converts stored pillar metadata into labeling_functions_v1 records:

- registry mechanisms are computed with the exact audited scorer code
  (clip-wide intervention scan, authority exact-span cues, creator-staging /
  WWYD source cues, instructional non-explanation polarity) and voted through
  their registry ``shadow_lf_vote_contract``;
- additional deterministic/model-assisted LFs are derived from fields the
  pipeline already stores (detector scene block, reaction texts, audio-event
  peaks, demo polarity, statement language priors);
- per-item eligibility gates are emitted alongside.

Everything is append-only shadow evidence.  No metadata, media, or label is
mutated, and no record is an acceptance decision.
"""

from __future__ import annotations

import re
from typing import Any

try:
    from weaksup.labeling_functions_v1 import (
        eligibility_gate,
        make_lf_record,
        registry_lf_records,
    )
    from scripts.score_witnessed_authority_reaction_cues import score_reaction
    from scripts.score_witnessed_staging_cues import score_text as staging_score_text
    from scripts.witnessed_reaction_candidate_scan import (
        candidate_scan_features,
        scan_reaction_candidates,
    )
    from scripts.witnessed_reaction_text_features import intervention_features
    from scripts.score_witnessed_reaction_candidate_windows import clip_segments
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from weaksup.labeling_functions_v1 import (
        eligibility_gate,
        make_lf_record,
        registry_lf_records,
    )
    from score_witnessed_authority_reaction_cues import score_reaction
    from score_witnessed_staging_cues import score_text as staging_score_text
    from witnessed_reaction_candidate_scan import (
        candidate_scan_features,
        scan_reaction_candidates,
    )
    from witnessed_reaction_text_features import intervention_features
    from score_witnessed_reaction_candidate_windows import clip_segments


MATERIALIZER_VERSION = "materialize_corpus_lf_records_v1"

AUDIO_PEAK_THRESHOLD = 0.2

NON_EXPLANATION_POLARITIES = {"contrast", "correct", "violation"}

# Transcript-language priors (aligned with score_transcript_scene_priors /
# score_transcript_visual_deixis; duplicated as frozen LF patterns so the LF
# version is stable even if the exploratory scripts evolve).
OCCURRED_EVENT = re.compile(
    r"\b(?:happened|occurred|saw|witnessed|caught on (?:camera|video|cctv)|"
    r"yesterday|last (?:week|night|year)|this (?:man|woman|guy|lady|person)\b.{0,40}\b"
    r"(?:did|was|got))\b",
    re.IGNORECASE,
)
REPORTED_OR_RETRO = re.compile(
    r"\b(?:said|says|told|reported|reportedly|according to|claimed|alleged(?:ly)?|"
    r"happened|recalled|had been)\b",
    re.IGNORECASE,
)
HYPOTHETICAL = re.compile(
    r"\b(?:imagine|what if|hypothetically|let'?s say|suppose|would you|"
    r"if (?:you|someone|somebody)|in general|people (?:who|that) (?:always|never))\b",
    re.IGNORECASE,
)
FOOTAGE_DEIXIS = re.compile(
    r"\b(?:you can see|as you can see|watch (?:this|as)|take a look|"
    r"(?:video|footage|clip) shows?|caught on (?:camera|video|cctv)|"
    r"in this (?:video|footage|clip))\b",
    re.IGNORECASE,
)
DEMO_TRANSITION = re.compile(
    r"\b(?:watch this|let'?s watch|look at this|here'?s an example|"
    r"role[- ]?play|demonstration|scenario|act it out|what happens)\b",
    re.IGNORECASE,
)


def _tri_record(
    *, item_id: str, pillar: str, target: str, lf_id: str, family: str,
    vote: int, abstain_reason: str | None, evidence: dict[str, Any],
) -> dict[str, Any]:
    return make_lf_record(
        item_id=item_id, pillar=pillar, target=target, lf_id=lf_id,
        family=family, vote=vote,
        abstain_reason=abstain_reason if vote == 0 else None,
        evidence=evidence, model_or_rule_version=MATERIALIZER_VERSION,
    )


def _reaction_texts(reactions: list[dict[str, Any]]) -> list[str]:
    texts = []
    for reaction in reactions:
        text = str(reaction.get("matched_text") or reaction.get("phrase") or "").strip()
        if text:
            texts.append(text)
    return texts


def witnessed_clip_items(metadata: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    by_clip: dict[int, list[dict[str, Any]]] = {}
    for reaction in metadata.get("reactions") or []:
        try:
            clip_idx = int(reaction["clip_idx"])
        except (KeyError, TypeError, ValueError):
            continue
        by_clip.setdefault(clip_idx, []).append(reaction)
    return by_clip


def witnessed_item_records(
    uid: str,
    metadata: dict[str, Any],
    transcript_segments: list[dict[str, Any]],
    registry: dict[str, Any],
    *,
    clip_exists: dict[int, bool] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """LF records + gate rows for every clip item of one witnessed source."""
    scene = ((metadata.get("provenance") or {}).get("scene") or {})
    audio = metadata.get("audio_events") or {}
    title = str(metadata.get("title") or "")
    channel = str(
        metadata.get("channel")
        or (metadata.get("source_meta") or {}).get("channel")
        or (metadata.get("source_meta") or {}).get("uploader")
        or ""
    )
    full_text = " ".join(
        str(segment.get("text") or "") for segment in transcript_segments
    )
    staging = staging_score_text(title, full_text, channel)

    records: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    for clip_idx, reactions in sorted(witnessed_clip_items(metadata).items()):
        item_id = f"witnessed:{uid}:clip_{clip_idx}"
        windows = [
            reaction["clip_window"]
            for reaction in reactions
            if isinstance(reaction.get("clip_window"), list)
            and len(reaction["clip_window"]) == 2
        ]
        # Registry rule signals for this item, computed by the audited code.
        scan_active = False
        if windows:
            start = min(float(w[0]) for w in windows)
            end = max(float(w[1]) for w in windows)
            segments = clip_segments(transcript_segments, start, end)
            selected_quote = " ".join(_reaction_texts(reactions))
            candidates = scan_reaction_candidates(segments, selected_quote=selected_quote)
            features = candidate_scan_features(candidates, selected_boundary=None)
            scan_active = features["candidate_scan.active_nonnegative"] > 0
        authority = any(
            score_reaction(reaction)["authority_reaction_cue"] for reaction in reactions
        )
        signal_outputs = {
            "witnessed_clipwide_intervention_scan_v1": scan_active,
            "witnessed_authority_exact_span_v3": authority,
            "witnessed_creator_staging_title_v2": bool(
                staging["witnessed_creator_staging_title_v2"]
            ),
            "witnessed_official_wwyd_channel_v1": bool(
                staging["witnessed_official_wwyd_channel_v1"]
            ),
        }
        records.extend(
            registry_lf_records(signal_outputs, registry, item_id=item_id, pillar="witnessed")
        )

        # Scene-block role (model-assisted; detector output stored in metadata).
        role = str(scene.get("reactor_role") or "")
        role_vote = {"bystander": 1, "victim": -1, "camera_person": -1}.get(role, 0)
        records.append(
            _tri_record(
                item_id=item_id, pillar="witnessed",
                target="independent_bystander_signal",
                lf_id="det_witnessed_scene_reactor_role_v1",
                family="actor_role_binding", vote=role_vote,
                abstain_reason="reactor_role_missing_or_mixed",
                evidence={"reactor_role": role or None, "provenance": "llm_scene_block"},
            )
        )
        # Head-count logic: <=1 visible person cannot include an independent
        # bystander; >=3 makes one possible (family distinct from role cues).
        n_people = scene.get("n_people")
        people_vote = 0
        if isinstance(n_people, (int, float)):
            people_vote = 1 if n_people >= 3 else -1 if n_people <= 1 else 0
        records.append(
            _tri_record(
                item_id=item_id, pillar="witnessed",
                target="independent_bystander_signal",
                lf_id="det_witnessed_scene_n_people_v1",
                family="social_context", vote=people_vote,
                abstain_reason="n_people_missing_or_two_person_scene",
                evidence={"n_people": n_people, "provenance": "llm_scene_block"},
            )
        )
        # Scene type: live action vs narration (described, not shown).
        scene_type = str(scene.get("scene_type") or "")
        scene_vote = {"action": 1, "narration": -1}.get(scene_type, 0)
        records.append(
            _tri_record(
                item_id=item_id, pillar="witnessed", target="norm_event_supported",
                lf_id="det_witnessed_scene_action_type_v1",
                family="event_action", vote=scene_vote,
                abstain_reason="scene_type_missing_or_mixed",
                evidence={"scene_type": scene_type or None, "provenance": "llm_scene_block"},
            )
        )
        # Reaction-language content: targeted objection vs generic affect.
        texts = _reaction_texts(reactions)
        features_by_text = [intervention_features(text) for text in texts]
        any_active = any(f["intervention.any_active"] for f in features_by_text)
        all_generic = bool(texts) and all(
            f["intervention.generic_affect_only"] for f in features_by_text
        )
        content_vote = 1 if any_active else -1 if all_generic else 0
        for target in ("norm_event_supported", "reaction_grounded"):
            records.append(
                _tri_record(
                    item_id=item_id, pillar="witnessed", target=target,
                    lf_id=f"det_witnessed_reaction_content_{target}_v1",
                    family="reaction", vote=content_vote,
                    abstain_reason="no_classifiable_reaction_text",
                    evidence={"n_reaction_texts": len(texts), "any_active": any_active,
                              "all_generic_affect": all_generic},
                )
            )
        # Nonverbal audio-event channel (PANNs summary stored in metadata).
        peaks = [audio.get("scream_peak"), audio.get("commotion_peak")]
        peak = max((float(p) for p in peaks if isinstance(p, (int, float))), default=None)
        records.append(
            _tri_record(
                item_id=item_id, pillar="witnessed", target="reaction_grounded",
                lf_id="det_witnessed_audio_event_peak_v1",
                family="reaction",
                vote=1 if peak is not None and peak >= AUDIO_PEAK_THRESHOLD else 0,
                abstain_reason="no_scored_audio_peak_above_threshold",
                evidence={"peak": peak, "threshold": AUDIO_PEAK_THRESHOLD},
            )
        )
        bounds_valid = bool(windows) and all(float(w[0]) < float(w[1]) for w in windows)
        gate_row = {
            "item_id": item_id,
            "media_decodes": (clip_exists or {}).get(clip_idx),
            "bounds_valid": bounds_valid,
        }
        gates.append({"item_id": item_id, **eligibility_gate(gate_row, "witnessed")})
    return records, gates


def instructional_item_records(
    uid: str,
    metadata: dict[str, Any],
    registry: dict[str, Any],
    *,
    demo_clip_exists: dict[int, bool] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """LF records + gates for every demo item of one instructional source."""
    records: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    for demo_idx, demo in enumerate(metadata.get("demos") or []):
        item_id = f"instructional:{uid}:demo_{demo_idx}"
        polarity = str(demo.get("polarity") or "")
        records.extend(
            registry_lf_records(
                {"instructional_non_explanation_review_priority_v1": polarity},
                registry, item_id=item_id, pillar="instructional",
            )
        )
        records.append(
            _tri_record(
                item_id=item_id, pillar="instructional", target="demonstration_present",
                lf_id="det_instructional_explanation_polarity_v1",
                family="explicit_semantics",
                vote=-1 if polarity == "explanation" else 0,
                abstain_reason="polarity_not_explanation",
                evidence={"polarity": polarity or None},
            )
        )
        demo_text = " ".join(str(value) for value in demo.values() if isinstance(value, str))
        transition = bool(DEMO_TRANSITION.search(demo_text))
        offscreen = bool(REPORTED_OR_RETRO.search(demo_text)) and not transition
        records.append(
            _tri_record(
                item_id=item_id, pillar="instructional", target="demonstration_present",
                lf_id="det_instructional_demo_transition_cue_v1",
                family="visual_depiction",
                vote=1 if transition else -1 if offscreen else 0,
                abstain_reason="no_transition_or_offscreen_language",
                evidence={"transition_cue": transition, "offscreen_language": offscreen},
            )
        )
        start, end = demo.get("start"), demo.get("end")
        has_bounds = isinstance(start, (int, float)) and isinstance(end, (int, float))
        gate_row = {
            "item_id": item_id,
            "media_decodes": (demo_clip_exists or {}).get(demo_idx),
            "bounds_valid": bool(has_bounds and float(start) < float(end))
            if has_bounds
            else None,
            "demonstration_interval_present": bool(
                has_bounds or (demo_clip_exists or {}).get(demo_idx)
            ),
        }
        gates.append({"item_id": item_id, **eligibility_gate(gate_row, "instructional")})
    return records, gates


COMMENTARY_CONTEXT_PAD_SEC = 20.0


def _statement_context(
    statement: dict[str, Any],
    transcript_segments: list[dict[str, Any]] | None,
    pad_sec: float = COMMENTARY_CONTEXT_PAD_SEC,
) -> tuple[str, str, bool]:
    """Return (quote, quote + surrounding transcript window, windowed?).

    Detectors store the stance quote; the event description usually lives in
    the transcript around it, so occurred/deixis evidence scans the window.
    """
    quote = " ".join(
        str(value) for value in statement.values() if isinstance(value, str)
    ) if isinstance(statement, dict) else str(statement)
    start = statement.get("start") if isinstance(statement, dict) else None
    end = statement.get("end") if isinstance(statement, dict) else None
    if not transcript_segments or not isinstance(start, (int, float)):
        return quote, quote, False
    window_start = float(start) - pad_sec
    window_end = float(end if isinstance(end, (int, float)) else start) + pad_sec
    parts = [
        str(segment.get("text") or "")
        for segment in transcript_segments
        if isinstance(segment.get("start"), (int, float))
        and float(segment.get("end") or segment["start"]) >= window_start
        and float(segment["start"]) <= window_end
    ]
    context = " ".join([quote] + parts).strip()
    return quote, context or quote, True


def commentary_item_records(
    uid: str,
    record: dict[str, Any],
    registry: dict[str, Any],
    *,
    transcript_segments: list[dict[str, Any]] | None = None,
    dual_vlm_positive: bool | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """LF records + gates for every statement of one commentary record.

    Text LFs (v2) scan a +-20 s transcript window around each statement, not
    just the stored stance quote.  Registry votes: the audited capture-title
    cue (computed from the record title) and, where a score exists, the
    dual-VLM retrieval core.
    """
    try:
        from scripts.score_commentary_title_event_cues import ANIMAL as TITLE_ANIMAL
        from scripts.score_commentary_title_event_cues import title_cues
    except ModuleNotFoundError:  # pragma: no cover - direct script execution
        from score_commentary_title_event_cues import ANIMAL as TITLE_ANIMAL
        from score_commentary_title_event_cues import title_cues

    title = str(record.get("title") or "")
    cues = title_cues(title) if title else []
    title_signal = bool(cues) and not TITLE_ANIMAL.search(title) and record.get(
        "agent"
    ) != "animal"
    signal_outputs: dict[str, Any] = {
        "commentary_capture_title_event_v1": title_signal,
    }
    if dual_vlm_positive is not None:
        signal_outputs["commentary_dual_vlm_retrieval_core_v1"] = dual_vlm_positive

    records: list[dict[str, Any]] = []
    gates: list[dict[str, Any]] = []
    for stmt_idx, statement in enumerate(record.get("statements") or []):
        item_id = f"commentary:{uid}:stmt_{stmt_idx}"
        records.extend(
            registry_lf_records(signal_outputs, registry, item_id=item_id, pillar="commentary")
        )
        quote, context, windowed = _statement_context(statement, transcript_segments)
        occurred = bool(
            OCCURRED_EVENT.search(context) or REPORTED_OR_RETRO.search(context)
        )
        # Hypothetical framing is judged on the quote itself; unrelated
        # "imagine if..." asides elsewhere in the window are not evidence
        # about this statement.
        hypothetical = bool(HYPOTHETICAL.search(quote))
        vote = -1 if hypothetical and not occurred else 1 if occurred else 0
        records.append(
            _tri_record(
                item_id=item_id, pillar="commentary", target="occurred_event_supported",
                lf_id="det_commentary_occurred_event_language_v2",
                family="event_action", vote=vote,
                abstain_reason="no_event_time_reference_in_window",
                evidence={"occurred_language": occurred, "hypothetical_language": hypothetical,
                          "windowed": windowed, "pad_sec": COMMENTARY_CONTEXT_PAD_SEC},
            )
        )
        deixis = bool(FOOTAGE_DEIXIS.search(context))
        records.append(
            _tri_record(
                item_id=item_id, pillar="commentary", target="occurred_event_supported",
                lf_id="det_commentary_footage_deixis_v2",
                family="visual_depiction",
                vote=1 if deixis else 0,
                abstain_reason="no_footage_deixis_in_window",
                evidence={"footage_deixis": deixis, "windowed": windowed},
            )
        )
        # Text target: eligibility is just a parseable statement quote.
        gates.append(
            {
                "item_id": item_id,
                "eligible": bool(quote.strip()),
                "failed_gates": [] if quote.strip() else ["statement_text_present"],
                "unknown_gates": [],
            }
        )
    return records, gates
