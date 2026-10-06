#!/usr/bin/env python3
"""Validate and combine manually audited weak-supervision signals.

The registry is an operational safety boundary.  A signal may retrieve, rank,
exclude from a strict route, or add a review route only when its frozen manual
audit authorizes that exact use.  No current rule creates a keep label.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


ALLOWED_USES = {
    "candidate_generation",
    "review_ranking",
    "strict_route_exclusion",
    "pillar_reroute",
    "shadow_lf_vote",
    "acceptance_gate",
}
NONDESTRUCTIVE_USES = ALLOWED_USES - {"acceptance_gate"}

SHADOW_LF_VOTES = (-1, 1)


def load_registry(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("registry must be a JSON object")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-9)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_evidence(
    rule_id: str,
    evidence: dict[str, Any],
    root: Path | None,
    *,
    require_source_disjoint: bool,
) -> None:
    artifact = evidence.get("artifact")
    if not isinstance(artifact, str) or not artifact:
        raise ValueError(f"{rule_id}: evidence artifact is required")
    if Path(artifact).is_absolute():
        raise ValueError(f"{rule_id}: evidence artifact must be repository-relative")
    expected_sha256 = evidence.get("artifact_sha256")
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError(f"{rule_id}: evidence artifact_sha256 is required")
    if root is not None:
        resolved = root / artifact
        if not resolved.is_file():
            raise ValueError(f"{rule_id}: missing evidence artifact: {artifact}")
        if _sha256(resolved) != expected_sha256:
            raise ValueError(f"{rule_id}: evidence artifact hash mismatch: {artifact}")
    if evidence.get("manual_review_complete") is not True:
        raise ValueError(f"{rule_id}: incomplete manual audit")
    if require_source_disjoint and evidence.get("source_disjoint") is not True:
        raise ValueError(f"{rule_id}: audit is not source-disjoint")
    if evidence.get("source_disjoint") not in {True, False}:
        raise ValueError(f"{rule_id}: source_disjoint must be explicit")
    reviewed = evidence.get("reviewed")
    if not isinstance(reviewed, int) or reviewed <= 0:
        raise ValueError(f"{rule_id}: evidence reviewed must be positive")

    tp, fp, fn = (evidence.get(name) for name in ("tp", "fp", "fn"))
    if all(value is None for value in (tp, fp, fn)):
        return
    if not all(isinstance(value, int) and value >= 0 for value in (tp, fp, fn)):
        raise ValueError(f"{rule_id}: tp/fp/fn must be nonnegative integers")
    selected = tp + fp
    if evidence.get("selected") != selected:
        raise ValueError(f"{rule_id}: selected does not equal tp + fp")
    expected_precision = tp / selected if selected else None
    expected_recall = tp / (tp + fn) if tp + fn else None
    for name, expected in (("precision", expected_precision), ("recall", expected_recall)):
        observed = evidence.get(name)
        if expected is None:
            if observed is not None:
                raise ValueError(f"{rule_id}: {name} must be null")
        elif not isinstance(observed, (int, float)) or not _close(float(observed), expected):
            raise ValueError(f"{rule_id}: inconsistent {name}")


def _validate_shadow_lf_vote_contract(rule_id: str, contract: Any) -> None:
    """A shadow LF vote is a new declared use: one target, one signed vote,
    one evidence family, and the audited error rates the label model may use.
    It affects no retrieval, route, acceptance, or media."""
    if not isinstance(contract, dict):
        raise ValueError(f"{rule_id}: shadow_lf_vote requires a contract object")
    for field in ("target", "evidence_family", "evidence_basis"):
        if not isinstance(contract.get(field), str) or not contract[field]:
            raise ValueError(f"{rule_id}: shadow_lf_vote_contract {field} is required")
    if contract.get("vote_when_triggered") not in SHADOW_LF_VOTES:
        raise ValueError(f"{rule_id}: vote_when_triggered must be -1 or 1")
    for metric in ("audited_precision", "audited_recall"):
        value = contract.get(metric)
        if value is not None and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            raise ValueError(f"{rule_id}: {metric} must be null or in [0, 1]")
    if contract.get("shadow_only") is not True:
        raise ValueError(f"{rule_id}: shadow_lf_vote_contract must set shadow_only true")


def validate_registry(registry: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    if registry.get("schema_version") != 1:
        raise ValueError("unsupported registry schema")
    rules = registry.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ValueError("registry rules must be a nonempty list")
    seen: set[str] = set()
    active = 0
    acceptance = 0
    for rule in rules:
        rule_id = rule.get("rule_id")
        if not isinstance(rule_id, str) or not rule_id or rule_id in seen:
            raise ValueError(f"invalid or duplicate rule_id: {rule_id!r}")
        seen.add(rule_id)
        uses = set(rule.get("allowed_uses") or [])
        if not uses <= ALLOWED_USES:
            raise ValueError(f"{rule_id}: unknown allowed use")
        if rule.get("automatic_acceptance") is not False:
            raise ValueError(f"{rule_id}: automatic_acceptance must be explicitly false")
        status = rule.get("status")
        evidence = rule.get("evidence") or []
        if not isinstance(evidence, list):
            raise ValueError(f"{rule_id}: evidence must be a list")
        for audit in evidence:
            _validate_evidence(
                rule_id,
                audit,
                root,
                require_source_disjoint=status == "audited_for_declared_use",
            )

        if status == "audited_for_declared_use":
            if not uses or not evidence:
                raise ValueError(f"{rule_id}: active rule lacks use or evidence")
            active += 1
        elif status == "failed_transfer":
            if uses:
                raise ValueError(f"{rule_id}: failed rule cannot have allowed uses")
        else:
            raise ValueError(f"{rule_id}: invalid status: {status!r}")

        if "shadow_lf_vote" in uses:
            _validate_shadow_lf_vote_contract(rule_id, rule.get("shadow_lf_vote_contract"))
        elif rule.get("shadow_lf_vote_contract") is not None:
            raise ValueError(f"{rule_id}: shadow_lf_vote_contract without declared use")

        if uses & NONDESTRUCTIVE_USES and "acceptance_gate" not in uses:
            if rule.get("trigger_values") in (None, []):
                raise ValueError(f"{rule_id}: active non-destructive rule lacks triggers")
        if "acceptance_gate" in uses:
            acceptance += 1
            contract = rule.get("promotion_contract") or {}
            if not (
                status == "audited_for_declared_use"
                and contract.get("target") == "final_pillar_acceptance"
                and int(contract.get("independent_confirmations") or 0) >= 2
                and int(contract.get("total_selected") or 0) >= 50
                and float(contract.get("precision") or 0) >= 0.90
                and float(contract.get("wilson_95_lower") or 0) >= 0.80
                and contract.get("recall") is not None
            ):
                raise ValueError(f"{rule_id}: acceptance promotion contract failed")

    if acceptance != int(registry.get("acceptance_gate_rules") or 0):
        raise ValueError("acceptance_gate_rules summary is inconsistent")
    return {
        "schema_version": 1,
        "rules": len(rules),
        "active_rules": active,
        "failed_transfer_rules": len(rules) - active,
        "acceptance_gate_rules": acceptance,
        "automatic_keep_enabled": acceptance > 0,
        "policy": "append_only_non_destructive_weak_supervision",
    }


def _triggered(value: Any, trigger_values: list[Any]) -> bool:
    return any(value == trigger for trigger in trigger_values)


def combine_signals(
    signal_outputs: dict[str, Any], registry: dict[str, Any]
) -> dict[str, Any]:
    """Combine audited outputs into review routes, never an implicit keep label."""
    validate_registry(registry)
    triggered: list[str] = []
    routes: set[str] = set()
    exclusions: set[str] = set()
    rankings: list[dict[str, Any]] = []
    unknown = sorted(set(signal_outputs) - {rule["rule_id"] for rule in registry["rules"]})
    for rule in registry["rules"]:
        rule_id = rule["rule_id"]
        if rule["status"] != "audited_for_declared_use" or rule_id not in signal_outputs:
            continue
        if not _triggered(signal_outputs[rule_id], rule.get("trigger_values") or []):
            continue
        triggered.append(rule_id)
        routes.update(rule.get("review_routes") or [])
        exclusions.update(rule.get("strict_route_exclusions") or [])
        if "review_ranking" in rule["allowed_uses"]:
            rankings.append(
                {
                    "rule_id": rule_id,
                    "priority_tier": rule.get("priority_tier", "standard"),
                }
            )
    return {
        "triggered_rule_ids": sorted(triggered),
        "review_routes": sorted(routes),
        "strict_route_exclusions": sorted(exclusions),
        "rankings": sorted(rankings, key=lambda row: row["rule_id"]),
        "unknown_signal_ids": unknown,
        "acceptance_label": None,
        "corpus_disposition": None,
        "delete_media": False,
        "policy": "audited_routes_only_no_automatic_keep_or_delete",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    result = validate_registry(registry, args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
