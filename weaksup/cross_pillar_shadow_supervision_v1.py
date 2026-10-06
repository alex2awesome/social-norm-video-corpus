#!/usr/bin/env python3
"""Correlation-aware shadow supervision for the three corpus pillars.

Inputs are atomic, enumerable observations from a human or model.  This module
does not inspect media and never creates an acceptance label.  It converts the
observations into auditable label-function votes, collapses correlated votes by
evidence family, and assigns only a manual-review priority band.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TRI = {"yes", "no", "uncertain"}
PILLARS = {"witnessed", "instructional", "commentary"}

FAMILY_FIELDS = {
    "witnessed": {
        "social_scope": ("social_actor_grounded", "concrete_behavior", "target_or_shared_context_grounded", "is_social_norm"),
        "distinct_reactor": ("distinct_bystander", "reactor_non_authority", "reactor_on_scene"),
        "causal_response": ("reaction_targets_event", "action_before_reaction"),
        "observable_reaction": ("reaction_observable",),
        "clean_training_span": ("pre_reaction_action_complete", "reaction_excluded_from_training_clip"),
        "organic_source": ("organic_authenticity",),
    },
    "instructional": {
        "social_scope": ("social_actor_grounded", "concrete_behavior", "target_or_shared_context_grounded", "is_social_norm"),
        "demonstration": ("demo_event_observable", "demo_action_complete"),
        "same_event_grounding": ("actor_action_target_same_event",),
        "semantic_alignment": ("demo_matches_extracted_norm",),
        "polarity_alignment": ("demo_matches_polarity",),
        "clean_training_span": ("demo_temporally_localized", "label_explanation_outside_training_clip"),
    },
    "commentary": {
        "social_scope": ("social_actor_grounded", "concrete_behavior", "target_or_shared_context_grounded", "is_social_norm"),
        "occurred_event_text": ("occurred_event_grounded", "behavior_semantically_specific", "normative_stance_grounded"),
        "same_event_grounding": ("actor_action_target_same_event",),
        "visual_localization": ("event_visible", "event_temporally_localized"),
        "semantic_alignment": ("visual_event_matches_repaired_label",),
        "clean_training_span": ("label_bearing_text_absent",),
    },
}

# Families are the unit of aggregation.  This prevents three observations made
# from one transcript phrase (for example actor/action/target) from masquerading
# as three independent Snorkel votes.
CORE_FAMILIES = {
    "witnessed": {"social_scope", "distinct_reactor", "causal_response", "observable_reaction", "clean_training_span", "organic_source"},
    "instructional": {"social_scope", "demonstration", "same_event_grounding", "semantic_alignment", "polarity_alignment", "clean_training_span"},
    "commentary": {"social_scope", "occurred_event_text", "same_event_grounding", "visual_localization", "semantic_alignment", "clean_training_span"},
}


def _family_vote(row: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    values = []
    for field in fields:
        value = row.get(field, "uncertain")
        if value not in TRI:
            raise ValueError(f"{field} must be yes/no/uncertain, got {value!r}")
        values.append(value)
    vote = "negative" if "no" in values else "positive" if all(v == "yes" for v in values) else "abstain"
    return {"vote": vote, "fields": dict(zip(fields, values))}


def score_record(row: dict[str, Any]) -> dict[str, Any]:
    pillar = row.get("pillar")
    if pillar not in PILLARS:
        raise ValueError(f"invalid pillar: {pillar!r}")
    families = {
        name: _family_vote(row, fields)
        for name, fields in FAMILY_FIELDS[pillar].items()
    }
    positive = sorted(k for k, v in families.items() if v["vote"] == "positive")
    negative = sorted(k for k, v in families.items() if v["vote"] == "negative")
    abstain = sorted(k for k, v in families.items() if v["vote"] == "abstain")
    core = CORE_FAMILIES[pillar]
    if negative:
        band = "likely_reroute_or_reject_review"
    elif core.issubset(positive):
        band = "complete_manual_review_candidate"
    elif len(core.intersection(positive)) >= max(2, len(core) - 2):
        band = "targeted_missing_evidence_review"
    else:
        band = "low_priority_preserved"
    return {
        **row,
        "shadow_policy_version": "cross_pillar_atomic_v1",
        "lf_families": families,
        "positive_families": positive,
        "negative_families": negative,
        "abstaining_families": abstain,
        "review_band": band,
        "automatic_acceptance": False,
        "corpus_disposition": None,
        "delete_media": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_jsonl", type=Path)
    parser.add_argument("output_jsonl", type=Path)
    args = parser.parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.input_jsonl.open() as src, args.output_jsonl.open("w") as dst:
        for line in src:
            if line.strip():
                dst.write(json.dumps(score_record(json.loads(line)), sort_keys=True) + "\n")


if __name__ == "__main__":
    main()

