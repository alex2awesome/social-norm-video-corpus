#!/usr/bin/env python3
"""One-shot, read-only staging of fresh audits from the live sk3 corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.export_instructional_demo_population_v1 import export_population
    from scripts.select_commentary_text_label_cohort_v2 import (
        MANUAL_FIELDS as COMMENTARY_FIELDS,
        select as select_commentary,
        transcript_context,
    )
    from scripts.sample_commentary_visual_audit import load_candidates, load_exclusions
    from scripts.select_instructional_temporal_critic_v4_fresh import (
        MANUAL_FIELDS as INSTRUCTIONAL_FIELDS,
        select as select_instructional,
    )
    from scripts.select_witnessed_role_causal_binding_v6_fresh import select as select_witnessed
    from scripts.select_witnessed_video_asr_corpus_audit import (
        read_jsonl, write_jsonl, write_manual_template,
    )
else:
    from export_instructional_demo_population_v1 import export_population
    from select_commentary_text_label_cohort_v2 import (
        MANUAL_FIELDS as COMMENTARY_FIELDS,
        select as select_commentary,
        transcript_context,
    )
    from sample_commentary_visual_audit import load_candidates, load_exclusions
    from select_instructional_temporal_critic_v4_fresh import (
        MANUAL_FIELDS as INSTRUCTIONAL_FIELDS,
        select as select_instructional,
    )
    from select_witnessed_role_causal_binding_v6_fresh import select as select_witnessed
    from select_witnessed_video_asr_corpus_audit import (
        read_jsonl, write_jsonl, write_manual_template,
    )


def first_json(path: Path) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    value = json.loads(line)
                    return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None
    return None


def discover_witnessed_inputs(root: Path) -> tuple[Path, Path]:
    search_roots = [
        root / "data" / "shadow_scores",
        root / "audit_runs" / "20260806_witnessed_video_asr_corpus_intensive_v1",
    ]
    manifests = []
    scores = []
    for search_root in search_roots:
        if not search_root.exists():
            continue
        for path in search_root.rglob("*.jsonl"):
            row = first_json(path)
            if row is None:
                continue
            if {
                "candidate_id", "item_id", "candidate_video_path"
            } <= set(row):
                manifests.append(path)
            if (
                "candidate_id" in row
                and ("result" in row or "raw_response" in row)
                and ("prompt_version" in row or "rubric" in row)
            ):
                scores.append(path)
    if not manifests or not scores:
        raise FileNotFoundError(
            f"witnessed inputs not found: manifests={manifests}, scores={scores}"
        )
    return max(manifests, key=lambda path: path.stat().st_size), max(
        scores, key=lambda path: path.stat().st_size
    )


def write_tsv(path: Path, fields: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    import csv
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def stage_instructional(root: Path, stage: Path, exclusions: Path) -> dict[str, Any]:
    population, population_summary = export_population(root / "data" / "instructional")
    population_path = stage / "instructional_population.jsonl"
    write_jsonl(population_path, population)
    (stage / "instructional_population_summary.json").write_text(
        json.dumps(population_summary, indent=2, sort_keys=True) + "\n"
    )
    excluded = {line.strip() for line in exclusions.read_text().splitlines() if line.strip()}
    selected = select_instructional(
        population, excluded, 100, "instructional-temporal-critic-v4-fresh-transfer"
    )
    out = stage / "instructional_v4"
    out.mkdir()
    semantic = []
    blind = []
    for index, row in enumerate(selected):
        candidate_id = f"instructional-v4-fresh-{index:04d}"
        semantic.append({**row, "audit_index": index, "candidate_id": candidate_id})
        blind.append({
            "audit_index": index, "candidate_id": candidate_id,
            "item_id": row["item_id"], "uid": row["uid"],
            "pillar": "instructional", "source_clip": row["source_clip"],
        })
    write_jsonl(out / "sealed_selection.jsonl", semantic)
    write_jsonl(out / "blind_source_manifest.jsonl", blind)
    write_tsv(out / "manual_gold.tsv", INSTRUCTIONAL_FIELDS, semantic)
    summary = {
        "selected": len(selected), "source_disjoint": True,
        "polarities": {polarity: sum(row["polarity"] == polarity for row in selected)
                       for polarity in sorted({row["polarity"] for row in selected})},
        "population": population_summary,
        "automatic_acceptance": False, "corpus_mutation_authorized": False,
    }
    (out / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def stage_commentary(root: Path, stage: Path, exclusions: Path) -> dict[str, Any]:
    candidates = load_candidates(
        root / "data" / "discussion", root / "data" / "discussion_video",
        load_exclusions(exclusions), "commentary-hierarchical-full-source-v2-text",
    )
    selected = select_commentary(
        candidates, 30, "commentary-hierarchical-full-source-v2-text"
    )
    out = stage / "commentary_text"
    out.mkdir()
    enriched = []
    for row in selected:
        value = dict(row)
        value["transcript_context"] = transcript_context(
            root / "data" / "transcripts" / f"{row['uid']}.json",
            float(row["detector_start_sec"]), float(row["detector_end_sec"]),
        )
        value["text_label_manual_reviewed"] = False
        value["script_certifies_visual_event"] = False
        enriched.append(value)
    write_jsonl(out / "sealed_text_selection.jsonl", enriched)
    write_tsv(out / "manual_text_review.tsv", COMMENTARY_FIELDS, enriched)
    summary = {
        "selected_sources": len(enriched), "source_disjoint": True,
        "platforms": {platform: sum(row["source_platform"] == platform for row in enriched)
                      for platform in sorted({row["source_platform"] for row in enriched})},
        "manual_text_review_required": True,
        "automatic_acceptance": False, "corpus_mutation_authorized": False,
    }
    (out / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def stage_witnessed(root: Path, stage: Path, exclusions: Path) -> dict[str, Any]:
    manifest, scores = discover_witnessed_inputs(root)
    excluded = {line.strip() for line in exclusions.read_text().splitlines() if line.strip()}
    sealed = select_witnessed(
        read_jsonl(manifest), read_jsonl(scores), excluded,
        uniform=80, enriched=120, seed="witnessed-role-causal-binding-v6-fresh",
    )
    out = stage / "witnessed_v6"
    out.mkdir()
    write_jsonl(out / "sealed_selection.jsonl", sealed)
    write_jsonl(out / "blind_visual_selection.jsonl", [{
        "clip_audit_index": row["clip_audit_index"], "item_id": row["item_id"],
        "uid": row["uid"], "candidates": row["candidates"],
    } for row in sealed])
    write_manual_template(out / "manual_atomic_ledger.tsv", sealed)
    summary = {
        "manifest": str(manifest), "v3_scores": str(scores),
        "selected_clips": len(sealed),
        "selected_candidates": sum(len(row["candidates"]) for row in sealed),
        "source_disjoint": True,
        "cohorts": {cohort: sum(row["cohort"] == cohort for row in sealed)
                    for cohort in sorted({row["cohort"] for row in sealed})},
        "reaction_target_ignores_staging": True,
        "automatic_acceptance": False, "corpus_mutation_authorized": False,
    }
    (out / "selection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    args = parser.parse_args()
    output_stage = args.stage / "outputs"
    output_stage.mkdir()
    report: dict[str, Any] = {
        "kind": "fresh_weak_supervision_cohort_staging",
        "root": str(args.root), "stage": str(args.stage),
        "automatic_acceptance": False, "corpus_mutation_authorized": False,
    }
    for name, function in (
        ("instructional", stage_instructional),
        ("commentary", stage_commentary),
        ("witnessed", stage_witnessed),
    ):
        try:
            report[name] = function(
                args.root, output_stage, args.exclusions / f"{name}_uids.txt"
            )
        except Exception as exc:
            report[name] = {"error": f"{type(exc).__name__}: {exc}"}
    (output_stage / "stage_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if all("error" not in report[name] for name in ("instructional", "commentary", "witnessed")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
