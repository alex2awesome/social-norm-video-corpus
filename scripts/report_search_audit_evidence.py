#!/usr/bin/env python3
"""Consolidate search and keep-rule evidence from immutable audit artifacts.

This reporter deliberately keeps three denominators separate:

1. query rank samples from explicit read-only search experiments;
2. manually judged corpus items from audit manifests;
3. reroutes/repairs, which are useful recovery evidence but not target-query
   precision.

It never reads or writes the production state database and never mutates corpus
metadata. The output is an append-only analysis artifact.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


RESULT_NAMES = (
    "manual_results.jsonl",
    "results_manual.jsonl",
    "manual_v2_results.jsonl",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected object")
        rows.append(value)
    return rows


def manifest_items(value: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(value, list):
        return {}, [row for row in value if isinstance(row, dict)]
    if not isinstance(value, dict):
        return {}, []
    rows = value.get("items") or value.get("records") or []
    return value, [row for row in rows if isinstance(row, dict)]


def accepted(pillar: str, result: dict[str, Any]) -> bool:
    decision = str(result.get("decision") or "")
    if pillar == "instructional":
        return decision in {"accept", "accept_after_trim", "accept_with_repairs"}
    if pillar == "witnessed":
        return decision == "accept_after_splice"
    if pillar == "commentary":
        return decision in {"accept", "accept_after_relabel"}
    return False


def evidence_axes(pillar: str, result: dict[str, Any]) -> dict[str, bool]:
    if pillar == "instructional":
        return {
            "is_social_norm": result.get("is_social_norm") == "yes",
            "visual_event": result.get("visual_demo_present") == "yes",
            "label_supported": result.get("norm_supported") == "yes",
            "strict_keep": accepted(pillar, result),
        }
    if pillar == "witnessed":
        return {
            "is_social_norm": result.get("is_social_norm") == "yes",
            "visual_event": result.get("social_action_visible") == "yes",
            "label_supported": result.get("behavior_label_supported") == "yes",
            "causal_reaction": (
                result.get("action_before_reaction") == "yes"
                and result.get("reaction_targets_action") == "yes"
                and result.get("reaction_is_normative") == "yes"
            ),
            "strict_keep": accepted(pillar, result),
        }
    if pillar == "commentary":
        return {
            "is_social_norm": result.get("is_social_norm") == "yes",
            "concrete_behavior": result.get("concrete_behavior") == "yes",
            "normative_stance": result.get("normative_stance_grounded") == "yes",
            "label_supported": result.get("proposed_norm_supported") == "yes",
            "strict_keep": accepted(pillar, result),
        }
    return {"strict_keep": False}


def grouped(rows: Iterable[dict[str, Any]], dimension: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        value = row.get(dimension)
        buckets[str(value) if value not in (None, "") else "NULL"].append(row)
    output = []
    for value, members in buckets.items():
        axes = sorted({axis for row in members for axis in row["axes"]})
        counts = {axis: sum(bool(row["axes"].get(axis)) for row in members) for axis in axes}
        output.append(
            {
                "value": value,
                "reviewed": len(members),
                "counts": counts,
                "rates": {axis: counts[axis] / len(members) for axis in axes},
            }
        )
    return sorted(output, key=lambda row: (-row["reviewed"], row["value"]))


def find_manifest(results_path: Path) -> Path | None:
    candidate = results_path.parent / "manifest.json"
    return candidate if candidate.is_file() else None


def artifact_stage(
    manifest: dict[str, Any], item: dict[str, Any], result_path: Path
) -> str:
    rubric = str(manifest.get("rubric_version") or "")
    batch_id = str(manifest.get("batch_id") or "")
    if (
        rubric == "instructional_recut_v1"
        or batch_id.startswith("recut_")
        or item.get("source_frame_manifest_sha256")
        or item.get("repair_type")
        or item.get("repaired_clip_path")
    ):
        return "repair"
    return "source"


def collect_manual_items(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    observations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for name in RESULT_NAMES:
        for result_path in sorted(root.glob(f"**/{name}")):
            manifest_path = find_manifest(result_path)
            if manifest_path is None:
                skipped.append({"results": str(result_path), "reason": "manifest_missing"})
                continue
            manifest, items = manifest_items(read_json(manifest_path))
            item_map = {
                str(item["item_id"]): item for item in items if item.get("item_id")
            }
            results = read_jsonl(result_path)
            for result in results:
                item_id = str(result.get("item_id") or "")
                item = item_map.get(item_id)
                if item is None:
                    skipped.append(
                        {
                            "results": str(result_path),
                            "item_id": item_id,
                            "reason": "item_not_in_manifest",
                        }
                    )
                    continue
                pillar = str(item.get("pillar") or manifest.get("pillar") or "")
                if pillar not in {"instructional", "witnessed", "commentary"}:
                    skipped.append(
                        {
                            "results": str(result_path),
                            "item_id": item_id,
                            "reason": "unknown_pillar",
                        }
                    )
                    continue
                observations.append(
                    {
                        "item_id": item_id,
                        "uid": item.get("uid"),
                        "pillar": pillar,
                        "category": item.get("category"),
                        "query_source": item.get("query_source"),
                        "found_by_query": item.get("found_by_query"),
                        "batch_id": result.get("batch_id") or manifest.get("batch_id"),
                        "selection_strategy": manifest.get("selection_strategy"),
                        "artifact_stage": artifact_stage(manifest, item, result_path),
                        "rubric_version": result.get("rubric_version"),
                        "model": result.get("model"),
                        "pass_index": int(result.get("pass_index") or 0),
                        "audited_at": float(result.get("audited_at") or 0),
                        "decision": result.get("decision"),
                        "axes": evidence_axes(pillar, result),
                        "manifest": str(manifest_path),
                        "results": str(result_path),
                    }
                )
    return observations, skipped


def canonical_latest(observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return one latest base judgment per item and pillar.

    Callers must select source or repair observations before invoking this
    function. This prevents an accepted recut from silently replacing the
    retrieval source outcome.
    """
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in observations:
        key = (row["pillar"], row["item_id"])
        order = (row["pass_index"], row["audited_at"], row["results"])
        prior = latest.get(key)
        if prior is None:
            latest[key] = row
            continue
        prior_order = (prior["pass_index"], prior["audited_at"], prior["results"])
        if order > prior_order:
            latest[key] = row
    return sorted(latest.values(), key=lambda row: (row["pillar"], row["item_id"]))


def collect_search_experiments(root: Path) -> list[dict[str, Any]]:
    experiments = []
    for path in sorted(root.glob("**/video_audit/manual_summary.json")):
        value = read_json(path)
        if not isinstance(value, dict) or not value.get("version"):
            continue
        version = str(value["version"])
        if "search_shadow" not in version:
            continue
        per_query = value.get("per_query") or []
        experiments.append(
            {
                "version": version,
                "path": str(path),
                "platform": value.get("platform", "dailymotion"),
                "selection": value.get("selection"),
                "enumerated_titles_reviewed": (
                    value.get("enumerated_titles_manually_reviewed")
                    or value.get("enumerated_titles_reviewed")
                ),
                "selected_sources": (
                    value.get("selected")
                    or value.get("selected_sources")
                    or 0
                ),
                "rendered_sources_reviewed": (
                    value.get("rendered_and_manually_reviewed")
                    or value.get("rendered_sources_reviewed")
                    or 0
                ),
                "production_eligible": bool(value.get("production_eligible")),
                "finding": value.get("finding"),
                "per_query": per_query,
            }
        )
    return experiments


def summarize(root: Path) -> dict[str, Any]:
    observations, skipped = collect_manual_items(root)
    source_observations = [
        row for row in observations if row["artifact_stage"] == "source"
    ]
    repair_observations = [
        row for row in observations if row["artifact_stage"] == "repair"
    ]
    canonical = canonical_latest(source_observations)
    canonical_repairs = canonical_latest(repair_observations)
    pillars: dict[str, Any] = {}
    for pillar in ("instructional", "witnessed", "commentary"):
        rows = [row for row in canonical if row["pillar"] == pillar]
        axes = sorted({axis for row in rows for axis in row["axes"]})
        pillars[pillar] = {
            "unique_source_items": len(rows),
            "axis_counts": {
                axis: sum(bool(row["axes"].get(axis)) for row in rows) for axis in axes
            },
            "by_query_source": grouped(rows, "query_source"),
            "by_category": grouped(rows, "category"),
            "by_found_by_query": grouped(rows, "found_by_query"),
            "repair_followups": {
                "unique_items": sum(
                    row["pillar"] == pillar for row in canonical_repairs
                ),
                "strict_keep": sum(
                    row["pillar"] == pillar and row["axes"].get("strict_keep")
                    for row in canonical_repairs
                ),
                "interpretation": (
                    "Recovery evidence only; never included in target-query "
                    "precision or source survival denominators."
                ),
            },
        }
    return {
        "schema_version": 1,
        "scope": str(root),
        "interpretation": {
            "manual_item_rates": (
                "Audited-set agreement only. Audit batches are often stratified or "
                "failure-enriched and do not estimate corpus prevalence."
            ),
            "search_experiment_rates": (
                "Only explicit top-rank read-only search experiments estimate "
                "retrieval yield; reroutes do not count as target-query precision."
            ),
            "query_or_title_is_label": False,
            "corpus_mutated": False,
        },
        "artifact_observations": len(observations),
        "canonical_unique_source_items": len(canonical),
        "canonical_unique_repair_items": len(canonical_repairs),
        "skipped": skipped,
        "pillars": pillars,
        "search_experiments": collect_search_experiments(root),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("audit_runs"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(args.audit_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "artifact_observations": report["artifact_observations"],
                "canonical_unique_source_items": report[
                    "canonical_unique_source_items"
                ],
                "canonical_unique_repair_items": report[
                    "canonical_unique_repair_items"
                ],
                "search_experiments": len(report["search_experiments"]),
                "skipped": len(report["skipped"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
