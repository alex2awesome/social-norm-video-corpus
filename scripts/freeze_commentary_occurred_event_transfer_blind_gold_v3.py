#!/usr/bin/env python3
"""Validate and freeze complete label-blind manual gold for commentary V3 transfer."""

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
    from scripts.commentary_occurred_event_contract_v3 import validate_result
    from scripts.select_commentary_occurred_event_transfer_v3 import read_jsonl
else:
    from commentary_occurred_event_contract_v3 import validate_result
    from select_commentary_occurred_event_transfer_v3 import read_jsonl


FORBIDDEN_REVEAL_FIELDS = {
    "uid", "item_id", "title", "query_source", "category", "signal",
    "v1_decision", "v1_normalized_behavior", "v1_normalized_norm",
}


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(
    packets: list[dict[str, Any]],
    gold: list[dict[str, Any]],
    observed_packet_sha256: str,
    expected_packet_sha256: str,
) -> dict[str, Any]:
    if observed_packet_sha256 != expected_packet_sha256:
        raise ValueError("blind packet hash differs from preregistration")
    packet_by_id = {str(row.get("blind_id") or ""): row for row in packets}
    gold_by_id = {str(row.get("blind_id") or ""): row for row in gold}
    if "" in packet_by_id or len(packet_by_id) != len(packets):
        raise ValueError("blind packet IDs are missing or duplicate")
    if "" in gold_by_id or len(gold_by_id) != len(gold):
        raise ValueError("manual gold IDs are missing or duplicate")
    if set(packet_by_id) != set(gold_by_id):
        raise ValueError("manual gold must exactly cover every blind packet")

    routes = Counter()
    strict = 0
    for blind_id, row in gold_by_id.items():
        if set(row) & FORBIDDEN_REVEAL_FIELDS:
            raise ValueError(f"{blind_id}: prior/retrieval metadata leaked into blind gold")
        if row.get("manual_reviewed") is not True or row.get("prior_label_revealed") is not False:
            raise ValueError(f"{blind_id}: blind manual-review attestations missing")
        packet = packet_by_id[blind_id]
        if row.get("transfer_index") != packet.get("transfer_index"):
            raise ValueError(f"{blind_id}: transfer index lineage mismatch")
        validated = validate_result(row)
        corpus = normalized(" ".join(
            str(segment.get("text") or "") for segment in packet["transcript_context"]
        ))
        for field in ("behavior_evidence_quote", "stance_evidence_quote"):
            quote = normalized(str(row.get(field) or ""))
            if quote and quote not in corpus:
                raise ValueError(f"{blind_id}: {field} is not grounded in blind transcript")
        routes[row["route"]] += 1
        strict += int(validated["strict_event_candidate"])
    return {
        "kind": "commentary_occurred_event_v3_transfer_blind_gold_freeze",
        "blind_packets": len(packets),
        "manually_reviewed": len(gold),
        "coverage": len(gold) / len(packets),
        "strict_event_candidates": strict,
        "strict_event_candidate_rate": strict / len(gold),
        "routes": dict(sorted(routes.items())),
        "blind_packet_sha256": observed_packet_sha256,
        "prior_labels_revealed": False,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--expected-packet-sha256", required=True)
    args = parser.parse_args()
    report = freeze(
        read_jsonl(args.packets),
        read_jsonl(args.gold),
        digest(args.packets),
        args.expected_packet_sha256,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
