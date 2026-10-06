#!/usr/bin/env python3
"""Add explicit relabel fields to fresh manual templates before review starts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = {
    "instructional": (
        "corrected_behavior_label", "source_audio_review_status",
        "demo_evidence_modalities",
    ),
    "commentary_blind": (
        "source_audio_review_status", "event_evidence_modalities",
    ),
    "commentary_post": ("corrected_behavior_label",),
    "witnessed": ("source_audio_review_status", "speaker_identity_basis"),
}


def upgrade(kind: str, source: Path, out: Path) -> dict[str, object]:
    if out.exists():
        raise FileExistsError(out)
    with source.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not reader.fieldnames:
            raise ValueError("manual template has no header")
        rows = list(reader)
        fields = list(reader.fieldnames)
    if not rows:
        raise ValueError("manual template is empty")
    additions = FIELDS[kind]
    if any(field in fields for field in additions):
        raise ValueError("manual template is already upgraded")
    fields.extend(additions)
    for row in rows:
        row.update({field: "" for field in additions})
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "kind": "fresh_manual_template_v2",
        "pillar_template": kind,
        "rows": len(rows),
        "added_fields": list(additions),
        "prior_judgments_present": any(
            value.strip()
            for row in rows
            for field, value in row.items()
            if field not in {"audit_index", "candidate_id", "item_id", "uid", "window_id"}
        ),
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=tuple(FIELDS), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(upgrade(args.kind, args.source, args.out), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
