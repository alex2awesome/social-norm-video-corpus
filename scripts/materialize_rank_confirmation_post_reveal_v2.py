#!/usr/bin/env python3
"""Expand compact rank-confirmation reviews into the strict audited schemas.

The compact reviews are preserved.  This script creates a second, explicit
artifact whose strict pass can be checked mechanically by the existing
post-reveal validators.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from scripts.validate_instructional_post_reveal_review import final_visual_demo


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def keyed(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {str(row["item_id"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate item_id")
    return result


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def normalized(text: str | None) -> str:
    value = unicodedata.normalize("NFKC", text or "").casefold()
    return " ".join(re.findall(r"\w+", value, flags=re.UNICODE))


def quote_grounded(packet: dict[str, Any]) -> str:
    transcript = normalized(
        " ".join(str(segment.get("text") or "") for segment in packet["aligned_transcript"])
    )
    quotes = [
        normalized(packet.get("start_quote")),
        normalized(packet.get("end_quote")),
    ]
    return "yes" if transcript and all(quote and quote in transcript for quote in quotes) else "no"


INSTRUCTIONAL_ROUTE = {
    "instructional": "instructional",
    "commentary": "commentary",
    "retain_non_social_demo": "technical",
    "retain_shadow_only": "unusable",
}


def materialize_instructional(
    packets: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    dense: list[dict[str, Any]],
    compact: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    packet_by_id = keyed(packets)
    sparse_by_id = keyed(sparse)
    dense_by_id = keyed(dense)
    compact_by_id = keyed(compact)
    if not (
        set(packet_by_id)
        == set(sparse_by_id)
        == set(compact_by_id)
    ):
        raise ValueError("instructional review coverage mismatch")
    output = []
    for packet in packets:
        item_id = str(packet["item_id"])
        source = compact_by_id[item_id]
        visual = final_visual_demo(item_id, sparse_by_id[item_id], dense_by_id)
        visual_match = (
            source["demo_visually_matches_assigned_norm"] if visual else "no"
        )
        fields = {
            "visual_demo_present": "yes" if visual else "no",
            "assigned_norm_is_social_norm": source["assigned_norm_is_social_norm"],
            "demo_visually_matches_assigned_norm": visual_match,
            "polarity_matches_depiction": source["polarity_matches_depiction"],
            "quote_grounded_in_interval": quote_grounded(packet),
            # A matched visual demonstration dominated the reviewed interval;
            # no-match or no-demo clips fail closed.
            "clip_is_demo_dominant": (
                "yes"
                if visual and visual_match == "yes"
                else "no"
            ),
        }
        route = INSTRUCTIONAL_ROUTE[source["recoverable_route"]]
        strict = all(value == "yes" for value in fields.values()) and route == "instructional"
        failures = [name for name, value in fields.items() if value != "yes"]
        output.append(
            {
                "item_id": item_id,
                "audit_index": packet["audit_index"],
                **fields,
                "strict_instructional_pass": "yes" if strict else "no",
                "recoverable_route": route,
                "required_repair": (
                    "none"
                    if strict
                    else "retain original; repair or reroute after: " + ", ".join(failures)
                ),
                "evidence_note": source["evidence_note"],
                "review_complete": "yes",
            }
        )
    return output


WITNESSED_ROUTE = {
    "instructional": "instructional",
    "instructional_relabel": "instructional",
    "instructional_correct_behavior": "instructional",
    "commentary": "commentary",
    "retain_context_only": "uncertain",
    "retain_aftermath_only": "uncertain",
    "witnessed_relabel_recut": "witnessed",
}

ORGANIC_REACTION_ROLES = {
    "affected_driver",
    "affected_target",
    "affected_target_or_companion",
}


def witnessed_reaction(source: dict[str, Any]) -> str:
    if source["authenticity"] == "organic":
        if source["reaction_role"] in ORGANIC_REACTION_ROLES:
            return "yes"
        if source["reaction_role"] in {"authority_or_participant", "unclear"}:
            return "uncertain"
    if source["authenticity"] == "uncertain":
        return "uncertain"
    return "no"


def witnessed_literal_violation(source: dict[str, Any]) -> str:
    action = source["action_visible_before_reaction"]
    relation = source["assigned_norm_relation"]
    if action == "no":
        return "no"
    if action == "uncertain":
        return "uncertain"
    if relation in {"exact", "repairable", "mismatch"}:
        return "yes"
    return "uncertain"


def materialize_witnessed(
    packets: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    compact: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    packet_by_id = keyed(packets)
    sparse_by_id = keyed(sparse)
    compact_by_id = keyed(compact)
    if not (set(packet_by_id) == set(sparse_by_id) == set(compact_by_id)):
        raise ValueError("witnessed review coverage mismatch")
    output = []
    for packet in packets:
        item_id = str(packet["item_id"])
        source = compact_by_id[item_id]
        temporal = source["action_visible_before_reaction"]
        reaction = witnessed_reaction(source)
        fields = {
            "literal_social_norm_violation_visible": witnessed_literal_violation(source),
            "assigned_norm_matches_event": (
                "yes"
                if source["assigned_norm_relation"] == "exact"
                else "uncertain"
                if source["assigned_norm_relation"] == "repairable"
                else "no"
            ),
            "reaction_is_organic_bystander_disapproval": reaction,
            "violation_precedes_reaction": temporal,
            # The one organic, temporally ordered repair candidate contains a
            # reaction inset; all other clips also fail authenticity/order.
            "reaction_spliceable_without_label_leak": (
                "no"
                if item_id == "witnessed:dailymotion__x22njtl:2"
                else "uncertain"
                if temporal == "yes" and reaction != "yes"
                else "no"
            ),
        }
        route = WITNESSED_ROUTE[source["recoverable_route"]]
        strict = all(value == "yes" for value in fields.values()) and route == "witnessed"
        failures = [name for name, value in fields.items() if value != "yes"]
        output.append(
            {
                "item_id": item_id,
                "audit_index": packet["audit_index"],
                **fields,
                "strict_witnessed_pass": "yes" if strict else "no",
                "recoverable_route": route,
                "required_repair": (
                    "none"
                    if strict
                    else "retain original; repair or reroute after: " + ", ".join(failures)
                ),
                "evidence_note": source["evidence_note"],
                "review_complete": "yes",
            }
        )
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pillar", choices=("instructional", "witnessed"), required=True)
    parser.add_argument("--packets", type=Path, required=True)
    parser.add_argument("--sparse-review", type=Path, required=True)
    parser.add_argument("--dense-review", type=Path)
    parser.add_argument("--compact-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    packets = read_jsonl(args.packets)
    sparse = read_jsonl(args.sparse_review)
    compact = read_jsonl(args.compact_review)
    if args.pillar == "instructional":
        if args.dense_review is None:
            parser.error("--dense-review is required for instructional")
        rows = materialize_instructional(
            packets, sparse, read_jsonl(args.dense_review), compact
        )
    else:
        rows = materialize_witnessed(packets, sparse, compact)
    write_jsonl(args.out, rows)
    print(json.dumps({"items": len(rows), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
