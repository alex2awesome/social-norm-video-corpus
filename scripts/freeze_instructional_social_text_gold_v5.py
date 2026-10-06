#!/usr/bin/env python3
"""Validate and freeze complete label-blind instructional social-text gold V5."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

if __package__:
    from scripts.instructional_social_behavior_contract_v5 import REQUIRED, validate_result
else:
    from instructional_social_behavior_contract_v5 import REQUIRED, validate_result


PACKET_FIELDS = {
    "blind_id",
    "audit_index",
    "proposed_norm",
    "proposed_polarity",
    "start_quote",
    "end_quote",
    "explanation",
}
GOLD_LINEAGE_FIELDS = {
    "blind_id",
    "audit_index",
    "manual_reviewed",
    "prior_visual_or_model_labels_revealed",
}
GOLD_FIELDS = REQUIRED | GOLD_LINEAGE_FIELDS
EVIDENCE_FIELDS = ("behavior_evidence_quote", "norm_evidence_quote")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def packet_evidence_corpus(packet: dict[str, Any]) -> str:
    return normalized(
        " ".join(
            str(packet.get(field) or "")
            for field in (
                "proposed_norm",
                "proposed_polarity",
                "start_quote",
                "end_quote",
                "explanation",
            )
        )
    )


def freeze(
    packets: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    observed_packet_sha256: str,
    expected_packet_sha256: str,
    observed_gold_sha256: str,
    *,
    expected_items: int = 100,
) -> dict[str, Any]:
    if observed_packet_sha256 != expected_packet_sha256:
        raise ValueError("blind packet hash differs from preregistration")
    if len(packets) != expected_items:
        raise ValueError(f"blind packet count must be exactly {expected_items}")
    if any(set(row) != PACKET_FIELDS for row in packets):
        raise ValueError("blind packet fields differ from sealed label-blind schema")

    packet_ids = [str(row.get("blind_id") or "") for row in packets]
    gold_ids = [str(row.get("blind_id") or "") for row in gold]
    if "" in packet_ids or len(set(packet_ids)) != len(packet_ids):
        raise ValueError("blind packet IDs are missing or duplicate")
    if "" in gold_ids or len(set(gold_ids)) != len(gold_ids):
        raise ValueError("manual gold IDs are missing or duplicate")
    if set(packet_ids) != set(gold_ids):
        raise ValueError("manual gold must exactly cover every blind packet")

    packet_by_id = {row["blind_id"]: row for row in packets}
    gold_by_id = {row["blind_id"]: row for row in gold}
    expected_indices = list(range(expected_items))
    if [row.get("audit_index") for row in packets] != expected_indices:
        raise ValueError("blind packet indices are incomplete or reordered")
    if sorted(row.get("audit_index") for row in gold) != expected_indices:
        raise ValueError("manual gold indices do not have exact unique coverage")

    routes: Counter[str] = Counter()
    domains: Counter[str] = Counter()
    strict = 0
    relabel = 0
    for blind_id in packet_ids:
        packet = packet_by_id[blind_id]
        row = gold_by_id[blind_id]
        unexpected = set(row) - GOLD_FIELDS
        missing = GOLD_FIELDS - set(row)
        if unexpected:
            raise ValueError(f"{blind_id}: reveal or unknown fields leaked into blind gold: {sorted(unexpected)}")
        if missing:
            raise ValueError(f"{blind_id}: manual gold fields missing: {sorted(missing)}")
        if row.get("manual_reviewed") is not True:
            raise ValueError(f"{blind_id}: manual-review attestation missing")
        if row.get("prior_visual_or_model_labels_revealed") is not False:
            raise ValueError(f"{blind_id}: visual/model evidence was revealed before freeze")
        if row.get("audit_index") != packet.get("audit_index"):
            raise ValueError(f"{blind_id}: audit index lineage mismatch")

        validated = validate_result(row)
        corpus = packet_evidence_corpus(packet)
        for field in EVIDENCE_FIELDS:
            quote = normalized(str(row.get(field) or ""))
            if quote and quote not in corpus:
                raise ValueError(f"{blind_id}: {field} is not grounded in the blind packet")
        routes[row["route"]] += 1
        domains[row["social_behavior_domain"]] += 1
        strict += int(validated["strict_text_candidate"])
        relabel += int(validated["text_candidate_after_relabel"])

    return {
        "kind": "instructional_social_behavior_v5_blind_text_gold_freeze",
        "blind_packets": len(packets),
        "manually_reviewed": len(gold),
        "coverage": len(gold) / len(packets),
        "strict_social_alignment": strict,
        "relabel_required": relabel,
        "literal_social_candidates_after_relabel": strict + relabel,
        "routes": dict(sorted(routes.items())),
        "domains": dict(sorted(domains.items())),
        "blind_packet_sha256": observed_packet_sha256,
        "blind_gold_sha256": observed_gold_sha256,
        "prior_visual_or_model_labels_revealed": False,
        "visual_demo_certified": False,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--expected-packet-sha256", required=True)
    parser.add_argument("--expected-items", type=int, default=100)
    args = parser.parse_args()
    report = freeze(
        read_jsonl(args.packets),
        read_jsonl(args.gold),
        digest(args.packets),
        args.expected_packet_sha256,
        digest(args.gold),
        expected_items=args.expected_items,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
