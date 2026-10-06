#!/usr/bin/env python3
"""Validate and apply a sealed manual commentary text review.

This is deliberately a text-only gate.  An accepted row says that the script
grounds an actor, concrete behavior, target/shared context, and normative
stance.  It never says that the behavior is visible.  The output is suitable
only for the later full-source visual-search packet builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any


ACCEPTED = {"accept", "accept_after_relabel"}
DECISIONS = ACCEPTED | {"reject"}
YES_NO = {"yes", "no"}
REQUIRED_FIELDS = (
    "uid",
    "item_id",
    "decision",
    "social_actor_grounded",
    "concrete_behavior",
    "target_or_shared_context_grounded",
    "normative_stance_grounded",
    "normalized_behavior",
    "normalized_norm",
    "behavior_evidence_quote",
    "stance_evidence_quote",
    "manual_rationale",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split())


def load_review(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != REQUIRED_FIELDS:
            raise ValueError("manual review has the wrong or reordered columns")
        return [{key: (value or "").strip() for key, value in row.items()} for row in reader]


def validate_and_merge(
    sealed: list[dict[str, Any]], review: list[dict[str, str]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not sealed:
        raise ValueError("sealed cohort is empty")
    sealed_by_item = {str(row.get("item_id") or ""): row for row in sealed}
    review_by_item = {row["item_id"]: row for row in review}
    if not all(sealed_by_item) or len(sealed_by_item) != len(sealed):
        raise ValueError("sealed cohort has missing or duplicate item_id")
    if not all(review_by_item) or len(review_by_item) != len(review):
        raise ValueError("manual review has missing or duplicate item_id")
    if set(sealed_by_item) != set(review_by_item):
        raise ValueError("manual review does not exactly cover sealed cohort")

    merged = []
    for item_id, source in sealed_by_item.items():
        gold = review_by_item[item_id]
        if gold["uid"] != str(source.get("uid") or ""):
            raise ValueError(f"{item_id}: uid changed after selection")
        if gold["decision"] not in DECISIONS:
            raise ValueError(f"{item_id}: invalid or missing decision")
        for field in (
            "social_actor_grounded",
            "concrete_behavior",
            "target_or_shared_context_grounded",
            "normative_stance_grounded",
        ):
            if gold[field] not in YES_NO:
                raise ValueError(f"{item_id}: {field} must be yes/no")
        if not gold["manual_rationale"]:
            raise ValueError(f"{item_id}: manual rationale is required")
        if gold["decision"] in ACCEPTED:
            for field in (
                "social_actor_grounded",
                "concrete_behavior",
                "target_or_shared_context_grounded",
                "normative_stance_grounded",
            ):
                if gold[field] != "yes":
                    raise ValueError(f"{item_id}: accepted row has {field}={gold[field]}")
            for field in (
                "normalized_behavior",
                "normalized_norm",
                "behavior_evidence_quote",
                "stance_evidence_quote",
            ):
                if not gold[field]:
                    raise ValueError(f"{item_id}: accepted row lacks {field}")
            context = normalize(" ".join(
                str(segment.get("text") or "")
                for segment in source.get("transcript_context") or []
            ))
            for field in ("behavior_evidence_quote", "stance_evidence_quote"):
                if normalize(gold[field]) not in context:
                    raise ValueError(f"{item_id}: {field} is not grounded in sealed context")

        row = dict(source)
        row.update(gold)
        row["text_label_manual_reviewed"] = True
        row["script_certifies_visual_event"] = False
        row["policy"] = "manual_text_gate_only_visual_evidence_still_required"
        merged.append(row)

    accepted = [row for row in merged if row["decision"] in ACCEPTED]
    counts = Counter(row["decision"] for row in merged)
    summary = {
        "kind": "commentary_manual_text_review_v2",
        "sealed_items": len(sealed),
        "manually_reviewed_items": len(merged),
        "coverage": len(merged) / len(sealed),
        "decisions": dict(sorted(counts.items())),
        "accepted_for_visual_search": len(accepted),
        "accepted_fraction": len(accepted) / len(merged),
        "script_certifies_visual_event": False,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }
    return merged, summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.out_dir.exists():
        raise FileExistsError(args.out_dir)
    merged, summary = validate_and_merge(read_jsonl(args.sealed), load_review(args.review))
    args.out_dir.mkdir(parents=True)
    write_jsonl(args.out_dir / "manual_text_gold.jsonl", merged)
    write_jsonl(
        args.out_dir / "accepted_visual_search_labels.jsonl",
        [row for row in merged if row["decision"] in ACCEPTED],
    )
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
