#!/usr/bin/env python3
"""Validate and seed manually audited ordinary-social-norm query canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

if __package__:
    from src import state
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src import state


AUDIT_FIELDS = {
    "query", "pillar", "manual_reviewed", "ordinary_social_norm",
    "police_confrontation", "specific_visible_target", "exact_prior_query",
    "decision", "risk",
}
PILLARS = {"instructional", "witnessed", "commentary"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate(
    plan: dict[str, Any],
    audit: list[dict[str, Any]],
    analysis: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    queries = plan.get("queries") or []
    if plan.get("policy", {}).get("police_or_law_enforcement_confrontation_allowed") is not False:
        raise ValueError("plan must explicitly prohibit police/law-enforcement confrontation")
    query_by_text = {str(row.get("query") or ""): row for row in queries}
    audit_by_text = {str(row.get("query") or ""): row for row in audit}
    if "" in query_by_text or len(query_by_text) != len(queries):
        raise ValueError("planned query strings are missing or duplicate")
    if "" in audit_by_text or len(audit_by_text) != len(audit):
        raise ValueError("manual query-audit strings are missing or duplicate")
    if set(query_by_text) != set(audit_by_text):
        raise ValueError("manual wording audit must exactly cover the query plan")

    for query, item in query_by_text.items():
        pillar = item.get("pillar")
        if pillar not in PILLARS or not str(item.get("category") or "").startswith(
            {"instructional": "instr_", "witnessed": "wit_", "commentary": "comm_"}[pillar]
        ):
            raise ValueError(f"{query}: invalid pillar/category lineage")
        if not item.get("platforms") or not set(item["platforms"]) <= {"youtube", "dailymotion", "reddit"}:
            raise ValueError(f"{query}: invalid platform list")
        if state.query_policy_reason(query, cfg):
            raise ValueError(f"{query}: blocked query theme")
        row = audit_by_text[query]
        if set(row) != AUDIT_FIELDS:
            raise ValueError(f"{query}: manual audit schema mismatch")
        if row["pillar"] != pillar:
            raise ValueError(f"{query}: pillar mismatch between plan and audit")
        if row["manual_reviewed"] is not True:
            raise ValueError(f"{query}: manual-review attestation missing")
        if any(row[field] != expected for field, expected in (
            ("ordinary_social_norm", "yes"),
            ("police_confrontation", "no"),
            ("specific_visible_target", "yes"),
            ("exact_prior_query", "no"),
        )):
            raise ValueError(f"{query}: wording audit did not pass")
        expected_decision = (
            "canary_approved" if item.get("canary")
            else "held_pending_canary_output_audit"
        )
        if row["decision"] != expected_decision or not str(row["risk"]).strip():
            raise ValueError(f"{query}: audit decision/risk mismatch")

    proposal = analysis.get("proposal") or {}
    if proposal.get("queries") != len(queries):
        raise ValueError("trajectory analysis does not cover the current plan")
    if proposal.get("exact_prior_query_count") != 0:
        raise ValueError("proposal contains a query already present in trajectory")
    if proposal.get("blocked_pattern_match_count") != 0:
        raise ValueError("proposal contains a blocked query theme")
    return {
        "queries": len(queries),
        "canaries": sum(bool(row.get("canary")) for row in queries),
        "by_pillar": dict(sorted(Counter(row["pillar"] for row in queries).items())),
        "manual_wording_audit_complete": True,
        "exact_prior_queries": 0,
        "blocked_theme_queries": 0,
    }


def seed(
    conn,
    plan: dict[str, Any],
    audit: list[dict[str, Any]],
    analysis: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    validated = validate(plan, audit, analysis, cfg)
    inserted = Counter()
    existing = Counter()
    source = str(plan.get("source") or "typical_social_norm_v1")
    for item in plan["queries"]:
        active = bool(item["canary"])
        for platform in item["platforms"]:
            added = state.add_query(
                conn, platform, item["query"], source,
                priority=3.25, category=item["category"], family=item["pillar"],
                active=active, cfg=cfg,
            )
            (inserted if added else existing)[item["pillar"]] += 1
    return {
        "kind": "typical_social_norm_query_v1_seed",
        **validated,
        "inserted_query_rows": sum(inserted.values()),
        "existing_query_rows": sum(existing.values()),
        "inserted_by_pillar": dict(sorted(inserted.items())),
        "active_scope": "manually_worded_canaries_only",
        "held_queries_active": False,
        "output_audit_required_before_expansion": True,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--manual-audit", type=Path, required=True)
    parser.add_argument("--trajectory-analysis", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--expected-analysis-sha256", required=True)
    args = parser.parse_args()
    if digest(args.plan) != args.expected_plan_sha256:
        raise ValueError("query plan hash differs from preregistration")
    if digest(args.trajectory_analysis) != args.expected_analysis_sha256:
        raise ValueError("trajectory analysis hash differs from preregistration")
    cfg = state.load_config(str(args.settings))
    conn = state.init_db(cfg)
    try:
        report = seed(
            conn, yaml.safe_load(args.plan.read_text()), read_jsonl(args.manual_audit),
            json.loads(args.trajectory_analysis.read_text()), cfg,
        )
    finally:
        conn.close()
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
