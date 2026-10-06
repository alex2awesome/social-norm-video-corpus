#!/usr/bin/env python3
"""Standardized labeling-function records for the shadow label model.

Every labeling function emits one immutable record per (item, target) with an
explicit ``+1``/``-1``/``0`` vote (roadmap section 12.3).  Failed-transfer
registry mechanisms abstain by construction, retrieval-only mechanisms
(query/title cues, review ranking) abstain because query or title text is never
a label, and correlated atomic observations are grouped into evidence families
so they cannot masquerade as independent votes.  No record mutates the corpus.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from weaksup.cross_pillar_shadow_supervision_v1 import FAMILY_FIELDS, _family_vote
    from weaksup.target_ontology_v1 import ONTOLOGY_VERSION, PILLARS, pillar_targets
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from weaksup.cross_pillar_shadow_supervision_v1 import FAMILY_FIELDS, _family_vote
    from weaksup.target_ontology_v1 import ONTOLOGY_VERSION, PILLARS, pillar_targets


LF_INTERFACE_VERSION = "labeling_functions_v1"

VOTES = (-1, 0, 1)

# Evidence families (roadmap section 12.5).  The label model treats votes in
# one family as dependent.
EVIDENCE_FAMILIES = (
    "event_action",
    "reaction",
    "temporal_causal_grounding",
    "actor_role_binding",
    "social_context",
    "explicit_semantics",
    "visual_depiction",
    "source_format_provenance",
    "contradiction_hard_negative",
    "semantic_model_judgment",
    "retrieval_lineage",
)

# Map the atomic-contract families of cross_pillar_shadow_supervision_v1 onto
# the shared evidence families above.
ATOMIC_FAMILY_MAP = {
    "social_scope": "social_context",
    "distinct_reactor": "actor_role_binding",
    "causal_response": "temporal_causal_grounding",
    "observable_reaction": "reaction",
    "clean_training_span": "contradiction_hard_negative",
    "organic_source": "source_format_provenance",
    "demonstration": "event_action",
    "same_event_grounding": "temporal_causal_grounding",
    "semantic_alignment": "explicit_semantics",
    "polarity_alignment": "explicit_semantics",
    "occurred_event_text": "event_action",
    "visual_localization": "visual_depiction",
}

# Default target each pillar's atomic-contract LFs vote on.
ATOMIC_DEFAULT_TARGET = {
    "witnessed": "norm_event_supported",
    "instructional": "demonstration_present",
    "commentary": "occurred_event_supported",
}

# Registry mechanisms vote only through an audited ``shadow_lf_vote_contract``
# in config/audited_weak_signals_v1.json.  The registry — not this module —
# is the source of truth for which mechanism may vote, on which target, with
# which sign and audited error rates.

# Deterministic eligibility gates (roadmap section 12.6).  A failed gate can
# never be outvoted by transcript evidence.
GATE_FIELDS: dict[str, tuple[str, ...]] = {
    "witnessed": ("media_decodes", "bounds_valid"),
    "instructional": ("media_decodes", "bounds_valid", "demonstration_interval_present"),
    "commentary": ("media_decodes", "bounds_valid", "event_interval_present"),
}

_HEX = set("0123456789abcdef")


def validate_lf_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("pillar") not in PILLARS:
        raise ValueError(f"invalid pillar: {record.get('pillar')!r}")
    if record.get("target") not in pillar_targets(record["pillar"]):
        raise ValueError(
            f"target {record.get('target')!r} is not a {record['pillar']} target"
        )
    for field in ("item_id", "lf_id", "model_or_rule_version"):
        if not isinstance(record.get(field), str) or not record[field]:
            raise ValueError(f"{field} must be a nonempty string")
    if record.get("family") not in EVIDENCE_FAMILIES:
        raise ValueError(f"invalid evidence family: {record.get('family')!r}")
    if record.get("vote") not in VOTES:
        raise ValueError(f"vote must be one of {VOTES}")
    reason = record.get("abstain_reason")
    if record["vote"] == 0:
        if not isinstance(reason, str) or not reason:
            raise ValueError("abstaining records require abstain_reason")
    elif reason is not None:
        raise ValueError("non-abstaining records must not carry abstain_reason")
    confidence = record.get("confidence")
    if confidence is not None and not (
        isinstance(confidence, (int, float)) and 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError("confidence must be null or in [0, 1]")
    if not isinstance(record.get("evidence"), dict):
        raise ValueError("evidence must be an object")
    sha = record.get("source_artifact_sha256")
    if sha is not None and not (
        isinstance(sha, str) and len(sha) == 64 and set(sha) <= _HEX
    ):
        raise ValueError("source_artifact_sha256 must be null or 64 hex chars")
    return record


def make_lf_record(
    *,
    item_id: str,
    pillar: str,
    target: str,
    lf_id: str,
    family: str,
    vote: int,
    confidence: float | None = None,
    abstain_reason: str | None = None,
    evidence: dict[str, Any] | None = None,
    model_or_rule_version: str = LF_INTERFACE_VERSION,
    source_artifact_sha256: str | None = None,
) -> dict[str, Any]:
    return validate_lf_record(
        {
            "item_id": item_id,
            "pillar": pillar,
            "target": target,
            "lf_id": lf_id,
            "family": family,
            "vote": vote,
            "confidence": confidence,
            "abstain_reason": abstain_reason,
            "evidence": evidence or {},
            "model_or_rule_version": model_or_rule_version,
            "source_artifact_sha256": source_artifact_sha256,
            "ontology_version": ONTOLOGY_VERSION,
        }
    )


def atomic_contract_lf_records(
    row: dict[str, Any], *, item_id: str, pillar: str, target: str | None = None
) -> list[dict[str, Any]]:
    """One LF per atomic-contract evidence family for the pillar's default
    target.  Unresolved atomic fields abstain rather than invent certainty."""
    if pillar not in PILLARS:
        raise ValueError(f"invalid pillar: {pillar!r}")
    resolved_target = target or ATOMIC_DEFAULT_TARGET[pillar]
    records = []
    for name, fields in FAMILY_FIELDS[pillar].items():
        family_result = _family_vote(row, fields)
        vote = {"positive": 1, "negative": -1, "abstain": 0}[family_result["vote"]]
        records.append(
            make_lf_record(
                item_id=item_id,
                pillar=pillar,
                target=resolved_target,
                lf_id=f"atomic_{pillar}_{name}_v1",
                family=ATOMIC_FAMILY_MAP[name],
                vote=vote,
                abstain_reason="unresolved_atomic_fields" if vote == 0 else None,
                evidence=family_result["fields"],
            )
        )
    return records


def registry_lf_records(
    signal_outputs: dict[str, Any],
    registry: dict[str, Any],
    *,
    item_id: str,
    pillar: str,
) -> list[dict[str, Any]]:
    """Wrap registered mechanisms as LF records.

    Failed-transfer rules abstain by construction.  Active rules without a
    registered ``shadow_lf_vote`` use abstain because their audits authorize
    retrieval or ordering, not labels.  A rule votes only when the registry
    carries an audited ``shadow_lf_vote_contract`` naming its target, sign,
    evidence family, and error rates — and it is triggered.
    """
    records = []
    for rule in registry["rules"]:
        rule_id = rule["rule_id"]
        if rule.get("pillar") not in {pillar, "all"} or rule_id not in signal_outputs:
            continue
        value = signal_outputs[rule_id]
        triggered = any(value == trigger for trigger in rule.get("trigger_values") or [])
        contract = rule.get("shadow_lf_vote_contract")
        may_vote = (
            rule.get("status") == "audited_for_declared_use"
            and "shadow_lf_vote" in (rule.get("allowed_uses") or [])
            and isinstance(contract, dict)
        )
        confidence = None
        if rule.get("status") != "audited_for_declared_use":
            vote, reason, target, family = (
                0,
                "failed_transfer_registry_enforced",
                ATOMIC_DEFAULT_TARGET[pillar],
                "semantic_model_judgment",
            )
        elif not may_vote:
            vote, reason, target, family = (
                0,
                "retrieval_or_ranking_use_only_never_a_label",
                ATOMIC_DEFAULT_TARGET[pillar],
                "retrieval_lineage",
            )
        elif not triggered:
            vote, reason, target, family = (
                0,
                "rule_not_triggered",
                contract["target"],
                contract["evidence_family"],
            )
        else:
            vote, reason, target, family = (
                contract["vote_when_triggered"],
                None,
                contract["target"],
                contract["evidence_family"],
            )
            confidence = contract.get("audited_precision")
        evidence = {"signal_value": value, "registry_status": rule.get("status")}
        if may_vote:
            evidence["audited_precision"] = contract.get("audited_precision")
            evidence["audited_recall"] = contract.get("audited_recall")
        records.append(
            make_lf_record(
                item_id=item_id,
                pillar=pillar,
                target=target,
                lf_id=f"registry_{rule_id}",
                family=family,
                vote=vote,
                confidence=confidence,
                abstain_reason=reason,
                evidence=evidence,
                model_or_rule_version=str(rule.get("version") or "audited_weak_signals_v1"),
            )
        )
    return records


def eligibility_gate(row: dict[str, Any], pillar: str) -> dict[str, Any]:
    """Deterministic gates that can never be outvoted (roadmap section 12.6)."""
    if pillar not in PILLARS:
        raise ValueError(f"invalid pillar: {pillar!r}")
    failed, unknown = [], []
    for gate in GATE_FIELDS[pillar]:
        value = row.get(gate)
        if value is True or value == "yes":
            continue
        if value is False or value == "no":
            failed.append(gate)
        else:
            unknown.append(gate)
    if row.get("sanitization_required") in (True, "yes") and row.get(
        "sanitization_verified"
    ) not in (True, "yes"):
        failed.append("sanitization_verified")
    return {
        "eligible": not failed and not unknown,
        "failed_gates": sorted(failed),
        "unknown_gates": sorted(unknown),
    }


def write_lf_records(records: list[dict[str, Any]], path) -> int:
    """Append-only JSONL writer; refuses to overwrite an existing shard."""
    if path.exists():
        raise FileExistsError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for record in records:
            handle.write(json.dumps(validate_lf_record(record), sort_keys=True) + "\n")
    return len(records)
