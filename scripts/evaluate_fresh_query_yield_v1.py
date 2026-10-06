#!/usr/bin/env python3
"""Measure search trajectories against the same fresh manual keeping gold.

Instructional and witnessed targets remain different.  Instructional yield is
a performed demo aligned to its weak label.  Witnessed retrieval yield is a
distinct bystander response; organic/social eligibility is reported as a
stricter downstream column rather than conflated with reaction retrieval.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

try:
    from scripts.witnessed_reaction_av_contract import candidate_positive
    from scripts.freeze_fresh_manual_audit_phase import validate_witnessed
except ModuleNotFoundError:
    from witnessed_reaction_av_contract import candidate_positive  # type: ignore[no-redef]
    from freeze_fresh_manual_audit_phase import validate_witnessed  # type: ignore[no-redef]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def indexed(rows: list[dict[str, Any]], field: str, name: str) -> dict[str, dict[str, Any]]:
    result = {str(row.get(field) or ""): row for row in rows}
    if not result or not all(result) or len(result) != len(rows):
        raise ValueError(f"{name} has missing or duplicate {field}")
    return result


def grouped(rows: list[dict[str, Any]], field: str, outcomes: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(field) or "<missing>")].append(row)
    output = {}
    for value, items in sorted(groups.items()):
        result: dict[str, Any] = {"reviewed_sources": len(items)}
        for outcome in outcomes:
            positives = sum(bool(row[outcome]) for row in items)
            result[outcome] = positives
            result[f"{outcome}_rate"] = positives / len(items)
        output[value] = result
    return output


def attributed(provenance: dict[str, Any]) -> dict[str, Any]:
    """Do not assign yield to one value when provenance sources conflict."""
    return {
        **provenance,
        "query_for_yield": (
            "<provenance_conflict>"
            if provenance.get("query_conflict") is True
            else provenance.get("query")
        ),
        "query_source_for_yield": (
            "<provenance_conflict>"
            if provenance.get("query_source_conflict") is True
            else provenance.get("query_source")
        ),
    }


def evaluate(
    provenance_rows: list[dict[str, Any]],
    instruction_rows: list[dict[str, Any]],
    instruction_gold_rows: list[dict[str, str]],
    witnessed_rows: list[dict[str, Any]],
    witnessed_gold_rows: list[dict[str, str]],
    witnessed_media_rows: list[dict[str, Any]],
    instruction_media_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    provenance = {
        (str(row.get("pillar") or ""), str(row.get("uid") or "")): row
        for row in provenance_rows
    }
    if len(provenance) != len(provenance_rows):
        raise ValueError("provenance has missing or duplicate pillar/uid")

    instruction_ids = [str(row.get("item_id") or "") for row in instruction_rows]
    instruction_uids = [str(row.get("uid") or "") for row in instruction_rows]
    witnessed_ids = [str(row.get("item_id") or "") for row in witnessed_rows]
    witnessed_uids = [str(row.get("uid") or "") for row in witnessed_rows]
    for name, ids, uids in (
        ("instructional", instruction_ids, instruction_uids),
        ("witnessed", witnessed_ids, witnessed_uids),
    ):
        if (
            any(not value for value in ids + uids)
            or len(ids) != len(set(ids))
            or len(uids) != len(set(uids))
        ):
            raise ValueError(f"{name} selection is not source-disjoint and unique")
    expected_provenance = {
        *(("instructional", uid) for uid in instruction_uids),
        *(("witnessed", uid) for uid in witnessed_uids),
    }
    if set(provenance) != expected_provenance:
        raise ValueError("provenance does not exactly cover selected sources")

    instruction_gold = indexed(
        instruction_gold_rows, "item_id", "instructional gold"
    )
    instruction_media = indexed(
        instruction_media_rows, "item_id", "instructional review media"
    )
    if set(instruction_gold) != set(instruction_ids):
        raise ValueError("instructional gold does not exactly cover selection")
    if set(instruction_media) != set(instruction_ids):
        raise ValueError("instructional review media does not exactly cover selection")
    instruction = []
    for source in instruction_rows:
        item_id, uid = str(source["item_id"]), str(source["uid"])
        if item_id not in instruction_gold or ("instructional", uid) not in provenance:
            raise ValueError(f"{item_id}: missing instructional gold or provenance")
        gold = instruction_gold[item_id]
        if gold.get("uid") not in {None, "", uid}:
            raise ValueError(f"{item_id}: instructional gold uid mismatch")
        expected_audio_status = (
            "reviewed" if instruction_media[item_id].get("audio_present") is True
            else "source_has_no_audio"
        )
        if gold.get("source_audio_review_status") != expected_audio_status:
            raise ValueError(f"{item_id}: instructional source audio was not reviewed")
        modalities = gold.get("demo_evidence_modalities")
        if (gold.get("visual_demo") == "no") != (modalities == "no_demo"):
            raise ValueError(f"{item_id}: instructional demo modalities contradict gold")
        if gold.get("visual_demo") not in {"yes", "no"}:
            raise ValueError(f"{item_id}: incomplete instructional manual gold")
        alignment = gold.get("label_alignment")
        if alignment not in {"exact", "partial", "mismatch", "no_visual"}:
            raise ValueError(f"{item_id}: incomplete instructional alignment")
        prov = provenance[("instructional", uid)]
        instruction.append({
            **attributed(prov),
            "visual_demo": gold["visual_demo"] == "yes",
            "clean_label_aligned_demo": (
                gold["visual_demo"] == "yes" and alignment == "exact"
            ),
            "demo_usable_after_relabel": (
                gold["visual_demo"] == "yes" and alignment in {"exact", "partial"}
            ),
        })

    witnessed_gold = indexed(
        witnessed_gold_rows, "candidate_id", "witnessed gold"
    )
    validate_witnessed(witnessed_rows, witnessed_gold_rows, witnessed_media_rows)
    expected_candidate_ids = {
        str(candidate.get("candidate_id") or "")
        for source in witnessed_rows
        for candidate in (source.get("candidates") or [])
    }
    if not expected_candidate_ids or "" in expected_candidate_ids or set(witnessed_gold) != expected_candidate_ids:
        raise ValueError("witnessed gold does not exactly cover selected candidates")
    witnessed = []
    for source in witnessed_rows:
        item_id, uid = str(source["item_id"]), str(source["uid"])
        candidates = source.get("candidates") or []
        ids = [str(row.get("candidate_id") or "") for row in candidates]
        if not ids or any(value not in witnessed_gold for value in ids):
            raise ValueError(f"{item_id}: missing witnessed candidate gold")
        if ("witnessed", uid) not in provenance:
            raise ValueError(f"{item_id}: missing witnessed provenance")
        labels = [witnessed_gold[value] for value in ids]
        if any(
            row.get("uid") not in {None, "", uid}
            or row.get("item_id") not in {None, "", item_id}
            for row in labels
        ):
            raise ValueError(f"{item_id}: witnessed gold lineage mismatch")
        distinct_reaction = any(candidate_positive(
            row, include_authority=False, require_social=False,
            require_unstaged=False,
        ) for row in labels)
        strict_organic_social = any(candidate_positive(
            row, include_authority=False, require_social=True,
            require_unstaged=True,
        ) for row in labels)
        witnessed.append({
            **attributed(provenance[("witnessed", uid)]),
            "cohort": source.get("cohort"),
            "distinct_bystander_reaction": distinct_reaction,
            "strict_organic_social_reaction": strict_organic_social,
        })

    instruction_outcomes = (
        "visual_demo", "clean_label_aligned_demo", "demo_usable_after_relabel",
    )
    witnessed_outcomes = (
        "distinct_bystander_reaction", "strict_organic_social_reaction",
    )
    uniform_witnessed = [
        row for row in witnessed if row.get("cohort") == "uniform_probability_sample"
    ]
    return {
        "kind": "fresh_query_yield_against_manual_gold_v1",
        "instructional": {
            "sources": len(instruction),
            "manual_source_audio_review_complete": True,
            "by_query_source": grouped(
                instruction, "query_source_for_yield", instruction_outcomes
            ),
            "by_exact_query": grouped(
                instruction, "query_for_yield", instruction_outcomes
            ),
        },
        "witnessed": {
            "sources": len(witnessed),
            "by_cohort": grouped(witnessed, "cohort", witnessed_outcomes),
            "population_query_yield": {
                "basis": "uniform_probability_sample_only",
                "sources": len(uniform_witnessed),
                "by_query_source": grouped(
                    uniform_witnessed, "query_source_for_yield", witnessed_outcomes
                ),
                "by_exact_query": grouped(
                    uniform_witnessed, "query_for_yield", witnessed_outcomes
                ),
            },
            "all_cohorts_descriptive_only": {
                "by_query_source": grouped(
                    witnessed, "query_source_for_yield", witnessed_outcomes
                ),
                "by_exact_query": grouped(
                    witnessed, "query_for_yield", witnessed_outcomes
                ),
                "not_a_population_yield_estimate": True,
            },
            "reaction_retrieval_is_not_conflated_with_organic_social_acceptance": True,
            "v3_positive_enrichment_excluded_from_population_query_yield": True,
            "manual_source_audio_review_complete": True,
        },
        "query_or_query_source_is_never_a_label": True,
        "automatic_search_change_authorized": False,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--instructional", type=Path, required=True)
    parser.add_argument("--instructional-gold", type=Path, required=True)
    parser.add_argument("--instructional-media", type=Path, required=True)
    parser.add_argument("--witnessed", type=Path, required=True)
    parser.add_argument("--witnessed-gold", type=Path, required=True)
    parser.add_argument("--witnessed-media", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    inputs = {
        "provenance": args.provenance,
        "instructional": args.instructional,
        "instructional_gold": args.instructional_gold,
        "instructional_media": args.instructional_media,
        "witnessed": args.witnessed,
        "witnessed_gold": args.witnessed_gold,
        "witnessed_media": args.witnessed_media,
    }
    report = evaluate(
        read_jsonl(args.provenance), read_jsonl(args.instructional),
        read_tsv(args.instructional_gold), read_jsonl(args.witnessed),
        read_tsv(args.witnessed_gold),
        read_jsonl(args.witnessed_media),
        read_jsonl(args.instructional_media),
    )
    report["artifacts_sha256"] = {key: sha256(path) for key, path in inputs.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
