#!/usr/bin/env python3
"""Build a frozen source-disjoint gold-cohort manifest and review ledgers.

Implements the roadmap section-13.4 audit sampling mixture against the revised
targets (target_ontology_v1) and the section-10.1 blind-before-reveal
methodology.  The output is a frozen selection manifest plus two empty ledger
templates: a blind ledger that deliberately carries no transcript, label,
query, or title information, and a reveal ledger completed only after the
blind visual judgment is frozen.  Selection is deterministic (seeded SHA-256
ordering), source-disjoint, duplicate-cluster-disjoint, and channel-capped.
Nothing is labeled, mutated, or deleted here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from weaksup.target_ontology_v1 import ONTOLOGY_VERSION, PILLARS
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from weaksup.target_ontology_v1 import ONTOLOGY_VERSION, PILLARS


MANIFEST_VERSION = "gold_cohort_manifest_v1"

# Section 13.4 mixture.  "uniform" is eligible for every row; the rest match
# tags carried on inventory rows by the upstream shadow-score exporters.
KNOWN_STRATA = (
    "uniform",
    "high_score",
    "low_score",
    "disagreement",
    "fresh_query",
    "repair_candidate",
)

# Blind ledger fields: atomic visual observations only, judged before any
# transcript/label/query reveal.  All revised targets are later derived from
# these atoms in code; the reviewer never fills a composite.
SHARED_BLIND_FIELDS = (
    "grounding_mode",
    "actor_kind",
    "behavior_kind",
    "affected_context_kind",
    "expectation_kind",
    "blind_description",
)
PILLAR_BLIND_FIELDS = {
    "witnessed": (
        "response_observable",
        "response_after_or_overlaps",
        "response_targets_action",
        "responder_role",
        "staging_apparent",
    ),
    "instructional": (
        "demo_event_observable",
        "demo_action_complete",
        "presentation_only",
    ),
    "commentary": (
        "event_visible",
        "event_temporally_localized",
    ),
}

# Reveal ledger fields: semantic judgment after the blind row is frozen.
REVEAL_FIELDS = (
    "label_relation",
    "polarity_matches",
    "normative_signal_grounded",
    "label_signal_excluded",
    "reroute",
    "repair_needed",
    "reveal_notes",
)


def _rank(seed: str, uid: str) -> str:
    return hashlib.sha256(f"{seed}:{uid}".encode()).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield row


def load_exclusions(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            uid = line.strip()
            if uid:
                excluded.add(uid)
    return excluded


def select_cohort(
    inventory: list[dict[str, Any]],
    *,
    seed: str,
    stratum_counts: dict[str, int],
    excluded_uids: set[str],
    channel_cap: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministic constrained selection.

    Strata are filled in declared order; every selected row consumes its
    source, duplicate cluster, and one channel slot globally, so no source,
    cluster, or over-capped channel can appear twice anywhere in the wave.
    """
    unknown = set(stratum_counts) - set(KNOWN_STRATA)
    if unknown:
        raise ValueError(f"unknown strata: {sorted(unknown)}")
    for row in inventory:
        for field in ("uid", "pillar", "source_uid"):
            if not row.get(field):
                raise ValueError(f"inventory row missing {field}: {row}")
        if row["pillar"] not in PILLARS:
            raise ValueError(f"invalid pillar: {row['pillar']!r}")

    ordered = sorted(inventory, key=lambda row: _rank(seed, row["uid"]))
    seen_uids: set[str] = set()
    seen_sources: set[str] = set()
    seen_clusters: set[str] = set()
    channel_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    shortfalls: dict[str, int] = {}

    for stratum, wanted in stratum_counts.items():
        taken = 0
        for row in ordered:
            if taken >= wanted:
                break
            uid = row["uid"]
            tags = row.get("strata") or []
            if stratum != "uniform" and stratum not in tags:
                continue
            channel = row.get("channel")
            cluster = row.get("cluster")
            if (
                uid in excluded_uids
                or uid in seen_uids
                or row["source_uid"] in seen_sources
                or (cluster is not None and cluster in seen_clusters)
                or (channel is not None and channel_counts.get(channel, 0) >= channel_cap)
            ):
                continue
            seen_uids.add(uid)
            seen_sources.add(row["source_uid"])
            if cluster is not None:
                seen_clusters.add(cluster)
            if channel is not None:
                channel_counts[channel] = channel_counts.get(channel, 0) + 1
            selected.append(
                {
                    **row,
                    "item_id": f"{row['pillar']}:{uid}",
                    "stratum": stratum,
                    "selection_rank": _rank(seed, uid),
                }
            )
            taken += 1
        if taken < wanted:
            shortfalls[stratum] = wanted - taken
    summary = {
        "manifest_version": MANIFEST_VERSION,
        "ontology_version": ONTOLOGY_VERSION,
        "seed": seed,
        "channel_cap": channel_cap,
        "requested": dict(stratum_counts),
        "selected_total": len(selected),
        "selected_by_stratum": {
            stratum: sum(row["stratum"] == stratum for row in selected)
            for stratum in stratum_counts
        },
        "selected_by_pillar": {
            pillar: sum(row["pillar"] == pillar for row in selected)
            for pillar in PILLARS
        },
        "shortfalls": shortfalls,
        "excluded_prior_uids": len(excluded_uids),
        "policy": "frozen_selection_no_labels_no_mutation",
    }
    return selected, summary


def blind_ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    """No transcript, label, query, title, or score reaches the blind pass."""
    template = {
        "item_id": row["item_id"],
        "pillar": row["pillar"],
        "media_ref": row.get("media_ref") or row["uid"],
        "reviewer": "",
        "blind_review_complete": False,
    }
    for field in SHARED_BLIND_FIELDS + PILLAR_BLIND_FIELDS[row["pillar"]]:
        template[field] = ""
    return template


def reveal_ledger_row(row: dict[str, Any]) -> dict[str, Any]:
    template = {
        "item_id": row["item_id"],
        "pillar": row["pillar"],
        "stratum": row["stratum"],
        "reviewer": "",
        "blind_row_frozen": False,
        "reveal_review_complete": False,
    }
    for field in REVEAL_FIELDS:
        template[field] = ""
    return template


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"output exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def build_outputs(
    selected: list[dict[str, Any]], summary: dict[str, Any], out_dir: Path
) -> dict[str, Any]:
    manifest_path = out_dir / "manifest.jsonl"
    _write_jsonl(selected, manifest_path)
    _write_jsonl([blind_ledger_row(row) for row in selected], out_dir / "blind_ledger.jsonl")
    _write_jsonl([reveal_ledger_row(row) for row in selected], out_dir / "reveal_ledger.jsonl")
    summary = {
        **summary,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    }
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        raise FileExistsError(f"output exists: {summary_path}")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def parse_stratum_args(pairs: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if not value or not value.isdigit() or int(value) <= 0:
            raise ValueError(f"invalid stratum spec: {pair!r} (want name=N)")
        if name in counts:
            raise ValueError(f"duplicate stratum: {name}")
        counts[name] = int(value)
    if not counts:
        raise ValueError("at least one --stratum is required")
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--exclude-uids", type=Path, action="append", default=[])
    parser.add_argument(
        "--stratum",
        action="append",
        default=[],
        metavar="NAME=N",
        help=f"stratum and count; known strata: {', '.join(KNOWN_STRATA)}",
    )
    parser.add_argument("--seed", required=True)
    parser.add_argument("--channel-cap", type=int, default=2)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    selected, summary = select_cohort(
        list(iter_jsonl(args.inventory)),
        seed=args.seed,
        stratum_counts=parse_stratum_args(args.stratum),
        excluded_uids=load_exclusions(args.exclude_uids),
        channel_cap=args.channel_cap,
    )
    summary = build_outputs(selected, summary, args.out_dir)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
