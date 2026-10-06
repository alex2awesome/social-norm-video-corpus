#!/usr/bin/env python3
"""Resolve a manually audited commentary repair chain to final outcomes.

Every initial candidate must end in exactly one audited pass or failure.  A
repair decision must have exactly one child in the next supplied stage, and a
passed artifact must satisfy the visible-action, label-leak, and clean-bounds
gates.  This produces a shadow ledger only; it never mutates corpus media.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique(rows: list[dict[str, Any]], key: str, name: str) -> dict[str, dict[str, Any]]:
    values = [str(row.get(key) or "") for row in rows]
    if not rows or any(not value for value in values) or len(set(values)) != len(values):
        raise ValueError(f"{name} is empty or has invalid/duplicate {key} values")
    return dict(zip(values, rows))


def _validate_decision(row: dict[str, Any], identity: str) -> None:
    outcome = str(row.get("outcome") or "")
    if not outcome.startswith(("pass_", "repair_", "fail_")):
        raise ValueError(f"{identity}: outcome must begin pass_, repair_, or fail_")
    for key in ("action_visible", "label_overlay_clean", "bounds_clean", "manual_evidence"):
        if not str(row.get(key) or "").strip():
            raise ValueError(f"{identity}: incomplete manual decision: {key}")
    if outcome.startswith("pass_") and tuple(
        str(row[key]) for key in ("action_visible", "label_overlay_clean", "bounds_clean")
    ) != ("yes", "yes", "yes"):
        raise ValueError(f"{identity}: passed artifact does not satisfy all visual gates")


def consolidate(
    initial_manifest: list[dict[str, Any]],
    initial_ledger: list[dict[str, Any]],
    stages: list[tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    initial_by_id = _unique(initial_manifest, "candidate_id", "initial manifest")
    initial_decisions = _unique(initial_ledger, "candidate_id", "initial ledger")
    if set(initial_by_id) != set(initial_decisions):
        raise ValueError("initial manifest and ledger cover different candidates")

    stage_maps: list[dict[str, tuple[dict[str, Any], dict[str, Any]]]] = []
    for stage_index, (plans, manifests, ledgers) in enumerate(stages, start=1):
        plan_by_id = _unique(plans, "variant_id", f"stage {stage_index} plan")
        manifest_by_id = _unique(manifests, "variant_id", f"stage {stage_index} manifest")
        ledger_by_id = _unique(ledgers, "variant_id", f"stage {stage_index} ledger")
        if set(plan_by_id) != set(manifest_by_id) or set(plan_by_id) != set(ledger_by_id):
            raise ValueError(f"stage {stage_index} plan, manifest, and ledger differ")
        parent_key = "parent_candidate_id" if stage_index == 1 else "parent_variant_id"
        children: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for variant_id, plan in plan_by_id.items():
            parent = str(plan.get(parent_key) or "")
            if not parent or parent in children:
                raise ValueError(f"stage {stage_index} has a missing or duplicate parent")
            children[parent] = (plan, ledger_by_id[variant_id])
        stage_maps.append(children)

    final_rows: list[dict[str, Any]] = []
    consumed_children: list[set[str]] = [set() for _ in stage_maps]
    for candidate_id, initial in initial_by_id.items():
        decision: dict[str, Any] = initial_decisions[candidate_id]
        identity = candidate_id
        chain = [{"stage": 0, "artifact_id": identity, "outcome": decision["outcome"]}]
        _validate_decision(decision, identity)
        for stage_index, children in enumerate(stage_maps, start=1):
            child = children.get(identity)
            needs_child = str(decision["outcome"]).startswith("repair_")
            if needs_child and child is None:
                raise ValueError(f"{identity}: unresolved repair before stage {stage_index}")
            if not needs_child and child is not None:
                raise ValueError(f"{identity}: non-repair decision has a repair child")
            if child is None:
                continue
            consumed_children[stage_index - 1].add(identity)
            plan, decision = child
            identity = str(plan["variant_id"])
            _validate_decision(decision, identity)
            chain.append(
                {"stage": stage_index, "artifact_id": identity, "outcome": decision["outcome"]}
            )
        if str(decision["outcome"]).startswith("repair_"):
            raise ValueError(f"{identity}: unresolved repair at end of chain")
        final_rows.append(
            {
                "candidate_id": candidate_id,
                "audit_index": int(initial["audit_index"]),
                "uid": initial.get("uid"),
                "final_artifact_id": identity,
                "final_outcome": decision["outcome"],
                "action_visible": decision["action_visible"],
                "label_overlay_clean": decision["label_overlay_clean"],
                "bounds_clean": decision["bounds_clean"],
                "manual_evidence": decision["manual_evidence"],
                "chain": chain,
                "acceptance_label": None,
                "corpus_disposition": None,
                "delete_media": False,
            }
        )

    for stage_index, children in enumerate(stage_maps):
        unused = set(children) - consumed_children[stage_index]
        if unused:
            raise ValueError(f"stage {stage_index + 1} contains orphan repair variants")

    outcomes = Counter(row["final_outcome"] for row in final_rows)
    passed = sum(row["final_outcome"].startswith("pass_") for row in final_rows)
    summary = {
        "kind": "commentary_manual_repair_chain_finalization",
        "policy": "shadow_only_preserve_all_source_media_no_automatic_acceptance",
        "initial_candidates": len(final_rows),
        "resolved_candidates": len(final_rows),
        "passed_visual_candidate_review": passed,
        "failed_visual_candidate_review": len(final_rows) - passed,
        "unresolved_repairs": 0,
        "pass_rate": passed / len(final_rows),
        "outcome_counts": dict(sorted(outcomes.items())),
        "all_source_media_preserved": True,
    }
    return final_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-manifest", type=Path, required=True)
    parser.add_argument("--initial-ledger", type=Path, required=True)
    parser.add_argument(
        "--stage",
        type=Path,
        nargs=3,
        action="append",
        metavar=("PLAN", "MANIFEST", "LEDGER"),
        default=[],
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.summary.exists():
        raise SystemExit("refusing to overwrite frozen output")
    stage_paths = args.stage or []
    rows, summary = consolidate(
        read_jsonl(args.initial_manifest),
        read_tsv(args.initial_ledger),
        [(read_jsonl(plan), read_jsonl(manifest), read_tsv(ledger)) for plan, manifest, ledger in stage_paths],
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary["evidence_sha256"] = {
        "initial_manifest": sha256(args.initial_manifest),
        "initial_ledger": sha256(args.initial_ledger),
        "stages": [
            {"plan": sha256(plan), "manifest": sha256(manifest), "ledger": sha256(ledger)}
            for plan, manifest, ledger in stage_paths
        ],
        "final_ledger": sha256(args.out),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
