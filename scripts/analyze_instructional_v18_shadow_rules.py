#!/usr/bin/env python3
"""Compare interpretable instructional shadow rules on frozen manual audits.

This is a development analysis only.  It joins already-computed model records
to source-disjoint V15 and V17 manual audits, reports transfer metrics, and
projects rule yield over the frozen V17 population.  It never mutates corpus
state or treats an unaudited model output as a keep/reject decision.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Callable

try:
    from scripts.evaluate_instructional_v11_holdout import metric, sha256
except ModuleNotFoundError:
    from evaluate_instructional_v11_holdout import metric, sha256


QUALIFYING_SCENES = {
    "situated_scene",
    "roleplay_demo",
    "animation_or_story_demo",
    "text_dialogue_demo",
}
PRESENTER_OR_BROLL = {
    "presenter_sample_or_advice",
    "retrospective_interview",
    "generic_broll_or_montage",
    "procedure_or_task_only",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def keyed_successes(path: Path) -> dict[str, dict[str, Any]]:
    """Return the last successful record for each item.

    Append-only VLM outputs may contain failed attempts followed by a retry.
    Failed attempts never overwrite an earlier or later success.
    """

    output: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("error") in (None, "") and isinstance(row.get("result"), dict):
            output[str(row["item_id"])] = row
    return output


def read_tsv(path: Path) -> dict[int, dict[str, str]]:
    with path.open() as handle:
        return {
            int(row["audit_index"]): row
            for row in csv.DictReader(handle, delimiter="\t")
        }


def yes(value: Any) -> bool:
    return str(value).strip().lower() in {"y", "yes", "true", "1"}


def result(record: dict[str, Any] | None) -> dict[str, Any]:
    if not record:
        return {}
    value = record.get("result")
    return value if isinstance(value, dict) else {}


def max_timestamp(source: dict[str, Any]) -> float:
    board = source.get("storyboard") or {}
    timestamps = board.get("sampled_timestamps") or []
    return float(max(timestamps, default=0.0))


def feature_record(
    source: dict[str, Any],
    glm_record: dict[str, Any] | None,
    qwen_record: dict[str, Any] | None,
    gemma_record: dict[str, Any] | None,
) -> dict[str, Any]:
    g = result(glm_record)
    q = result(qwen_record)
    m = result(gemma_record)
    q_scene = str(q.get("scene_role", ""))
    g_scene = str(g.get("scene_role", ""))
    q_scope = str(q.get("social_scope", ""))
    quote_role = str(m.get("quote_role", ""))
    visual_score = (
        4 * (g.get("demo_usable") == "yes")
        + 2 * (g_scene == "animation_or_story_demo")
        + 2 * (q_scene in QUALIFYING_SCENES)
        + 1 * (q.get("performed_social_behavior") == "yes")
        + 1 * (q.get("affected_party_or_shared_setting_present") == "yes")
        + 2 * (m.get("visual_record_supports_concrete_demo") == "yes")
        + 1 * (m.get("same_actor_and_target") == "yes")
        + 1 * (quote_role == "performed_dialogue")
        - 3 * (q_scene in PRESENTER_OR_BROLL)
        - 2 * (g_scene in PRESENTER_OR_BROLL)
        - 1 * (max_timestamp(source) > 90)
    )
    return {
        "item_id": source["item_id"],
        "uid": source["uid"],
        "audit_index": source.get("audit_index"),
        "band": source.get("band"),
        "glm_demo": g.get("demo_usable") == "yes",
        "glm_animation": g_scene == "animation_or_story_demo",
        "glm_scene": g_scene in QUALIFYING_SCENES,
        "qwen_scene": q_scene in QUALIFYING_SCENES,
        "qwen_presenter_or_broll": q_scene in PRESENTER_OR_BROLL,
        "qwen_performed": q.get("performed_social_behavior") == "yes",
        "qwen_socially_evaluable": (
            q.get("socially_evaluable_without_metadata") == "yes"
        ),
        "qwen_target_performed": (
            q.get("target_behavior_performed_not_described") == "yes"
        ),
        "qwen_response": (
            q.get("social_response_or_consequence_present") == "yes"
        ),
        "qwen_social_scope": q_scope,
        "qwen_tacit_or_etiquette": q_scope
        in {"tacit_interpersonal", "explicit_social_etiquette"},
        "qwen_affected": (
            q.get("affected_party_or_shared_setting_present") == "yes"
        ),
        "gemma_visual": (
            m.get("visual_record_supports_concrete_demo") == "yes"
        ),
        "gemma_actor_target": m.get("same_actor_and_target") == "yes",
        "gemma_performed_dialogue": quote_role == "performed_dialogue",
        "gemma_depicted_quote": quote_role
        in {"performed_dialogue", "narrated_depicted_action"},
        "gemma_causal": m.get("causal_or_intent_match")
        in {"yes", "not_required"},
        "gemma_polarity": (
            m.get("proposed_violation_polarity_matches") == "yes"
        ),
        "gemma_exact_norm": m.get("norm_relation") == "exact",
        "duration_over_90s": max_timestamp(source) > 90,
        "visual_score": int(visual_score),
    }


Rule = Callable[[dict[str, Any]], bool]


RULES: dict[str, Rule] = {
    "glm_demo": lambda row: row["glm_demo"],
    "glm_animation": lambda row: row["glm_demo"] and row["glm_animation"],
    "glm_animation_qwen_social": lambda row: (
        row["glm_demo"]
        and row["glm_animation"]
        and row["qwen_performed"]
        and row["qwen_socially_evaluable"]
    ),
    "glm_animation_qwen_social_response": lambda row: (
        row["glm_demo"]
        and row["glm_animation"]
        and row["qwen_performed"]
        and row["qwen_socially_evaluable"]
        and row["qwen_response"]
    ),
    "glm_animation_qwen_tacit_or_etiquette": lambda row: (
        row["glm_demo"]
        and row["glm_animation"]
        and row["qwen_performed"]
        and row["qwen_socially_evaluable"]
        and row["qwen_tacit_or_etiquette"]
    ),
    "glm_animation_qwen_social_gemma_exact": lambda row: (
        row["glm_demo"]
        and row["glm_animation"]
        and row["qwen_performed"]
        and row["qwen_socially_evaluable"]
        and row["gemma_exact_norm"]
        and row["gemma_polarity"]
    ),
    "glm_qwen_scene": lambda row: row["glm_demo"] and row["qwen_scene"],
    "glm_qwen_scene_gemma_visual": lambda row: (
        row["glm_demo"] and row["qwen_scene"] and row["gemma_visual"]
    ),
    "performed_dialogue_tier": lambda row: (
        row["glm_demo"]
        and row["qwen_scene"]
        and row["gemma_visual"]
        and row["gemma_actor_target"]
        and row["gemma_performed_dialogue"]
    ),
    "animation_or_performed_dialogue": lambda row: (
        row["glm_demo"]
        and row["qwen_scene"]
        and row["gemma_visual"]
        and (
            row["glm_animation"]
            or (
                row["gemma_actor_target"]
                and row["gemma_performed_dialogue"]
            )
        )
    ),
    "visual_score_ge_8": lambda row: row["visual_score"] >= 8,
    "visual_score_ge_10": lambda row: row["visual_score"] >= 10,
    "visual_score_ge_12": lambda row: row["visual_score"] >= 12,
}


def audit_rows(
    semantic_path: Path,
    visual_path: Path,
    manual_semantic_path: Path,
    glm_path: Path | None = None,
    qwen_path: Path | None = None,
    gemma_path: Path | None = None,
) -> list[dict[str, Any]]:
    semantic = read_jsonl(semantic_path)
    visual = read_tsv(visual_path)
    manual = read_tsv(manual_semantic_path)
    glm = keyed_successes(glm_path) if glm_path else {}
    qwen = keyed_successes(qwen_path) if qwen_path else {}
    gemma = keyed_successes(gemma_path) if gemma_path else {}
    rows: list[dict[str, Any]] = []
    for source in semantic:
        index = int(source["audit_index"])
        item_id = str(source["item_id"])
        opaque_id = str(source.get("candidate_id", ""))
        features = feature_record(
            source,
            source.get("glm_v10a")
            or source.get("v10a")
            or glm.get(item_id)
            or glm.get(opaque_id),
            source.get("qwen_v16")
            or qwen.get(item_id)
            or qwen.get(opaque_id),
            source.get("gemma_causal")
            or gemma.get(item_id)
            or gemma.get(opaque_id),
        )
        features.update(
            {
                "visual": yes(visual[index]["visual_demo"]),
                "exact": yes(
                    manual[index].get("exact_original")
                    or manual[index].get("exact_social_norm")
                ),
                "relabel": yes(
                    manual[index].get("usable_after_relabel")
                    or manual[index].get("relabel_usable")
                ),
            }
        )
        rows.append(features)
    return rows


def target_metrics(
    rows: list[dict[str, Any]],
    accepted: list[bool],
    target: str,
) -> dict[str, Any]:
    selected = [row for row, keep in zip(rows, accepted) if keep]
    value = metric(selected, lambda row: bool(row[target]))
    value["accepted"] = len(selected)
    value["recall_in_audit"] = (
        sum(bool(row[target]) for row in selected)
        / sum(bool(row[target]) for row in rows)
        if any(bool(row[target]) for row in rows)
        else None
    )
    value["accepted_indices"] = [
        row["audit_index"] for row in selected
    ]
    value["false_positive_indices"] = [
        row["audit_index"] for row in selected if not row[target]
    ]
    return value


def evaluate_rules(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, rule in RULES.items():
        accepted = [bool(rule(row)) for row in rows]
        output[name] = {
            target: target_metrics(rows, accepted, target)
            for target in ("visual", "exact", "relabel")
        }
    return output


def weighted_metrics(
    rows: list[dict[str, Any]],
    rule: Rule,
    target: str,
    population_by_band: dict[str, int],
) -> dict[str, Any]:
    selected_weight = 0.0
    positive_weight = 0.0
    total_positive_weight = 0.0
    sample_by_band = Counter(str(row["band"]) for row in rows)
    for row in rows:
        band = str(row["band"])
        weight = population_by_band[band] / sample_by_band[band]
        if row[target]:
            total_positive_weight += weight
        if rule(row):
            selected_weight += weight
            if row[target]:
                positive_weight += weight
    return {
        "estimated_accepted": selected_weight,
        "estimated_positive": positive_weight,
        "estimated_precision": (
            positive_weight / selected_weight if selected_weight else None
        ),
        "estimated_recall": (
            positive_weight / total_positive_weight
            if total_positive_weight
            else None
        ),
        "warning": (
            "Inverse-band-weighted development estimate; small within-band "
            "cells are not a substitute for a fresh holdout."
        ),
    }


def population_rows(
    source_path: Path,
    storyboard_path: Path,
    glm_path: Path,
    qwen_path: Path,
    gemma_path: Path,
    excluded_uids: set[str],
) -> list[dict[str, Any]]:
    storyboards = {
        str(row["item_id"]): row for row in read_jsonl(storyboard_path)
    }
    glm = keyed_successes(glm_path)
    qwen = keyed_successes(qwen_path)
    gemma = keyed_successes(gemma_path)
    rows: list[dict[str, Any]] = []
    for source in read_jsonl(source_path):
        if str(source["uid"]) in excluded_uids:
            continue
        item_id = str(source["item_id"])
        if item_id not in glm or item_id not in qwen or item_id not in gemma:
            raise ValueError(f"missing complete model triplet: {item_id}")
        if item_id not in storyboards:
            raise ValueError(f"missing storyboard record: {item_id}")
        source = {**source, "storyboard": storyboards[item_id]}
        rows.append(
            feature_record(
                source, glm[item_id], qwen[item_id], gemma[item_id]
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("audit_runs"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "audit_runs/20260729_instructional_v17_causal_transfer_v1/"
            "v18_shadow_rule_analysis.json"
        ),
    )
    args = parser.parse_args()
    v15_dir = (
        args.audit_root
        / "20260728_instructional_v15_utterance_anchor_prospective_v1"
    )
    v17_dir = (
        args.audit_root
        / "20260729_instructional_v17_causal_transfer_v1"
    )
    v18_dir = (
        args.audit_root
        / "20260729_instructional_v18_animation_holdout_v1"
    )
    full = v17_dir / "full_population"
    v15 = audit_rows(
        v15_dir / "audit_selection_semantic.jsonl",
        v15_dir / "manual_visual_ledger_blind.tsv",
        v15_dir / "manual_semantic_ledger.tsv",
        qwen_path=v15_dir / "qwen_v16.jsonl",
        gemma_path=v15_dir / "gemma_v12_causal.jsonl",
    )
    v17 = audit_rows(
        v17_dir / "audit_selection_semantic.jsonl",
        v17_dir / "manual_visual_ledger_final.tsv",
        v17_dir / "manual_semantic_ledger.tsv",
    )
    v18 = audit_rows(
        v18_dir / "audit_selection_semantic.jsonl",
        v18_dir / "manual_visual_ledger_blind.tsv",
        v18_dir / "manual_semantic_ledger.tsv",
    )
    prior_uids = {str(row["uid"]) for row in v15}
    population = population_rows(
        full / "source_population.jsonl",
        full / "storyboard_manifest.jsonl",
        full / "glm_v10a.jsonl",
        full / "qwen_v16_full.jsonl",
        full / "gemma_v12_causal_full.jsonl",
        prior_uids,
    )
    if len(population) != 1370:
        raise ValueError(f"expected 1370 eligible rows, got {len(population)}")
    all_audited_uids = {
        str(row["uid"]) for row in [*v15, *v17, *v18]
    }
    v19_population = population_rows(
        full / "source_population.jsonl",
        full / "storyboard_manifest.jsonl",
        full / "glm_v10a.jsonl",
        full / "qwen_v16_full.jsonl",
        full / "gemma_v12_causal_full.jsonl",
        all_audited_uids,
    )
    if len(v19_population) != 1263:
        raise ValueError(
            f"expected 1263 source-disjoint V19 rows, got {len(v19_population)}"
        )

    band_counts = {
        "candidate": 2,
        "gemma_causal_reject": 44,
        "glm_v10a_reject": 1057,
        "qwen_v16_reject": 267,
    }
    report: dict[str, Any] = {
        "kind": "instructional_v18_interpretable_shadow_rule_development",
        "status": "posthoc_development_not_promotion",
        "policy": "shadow ranking only; no automatic keep, reject, or mutation",
        "coverage": {
            "v15_source_disjoint_development": len(v15),
            "v17_source_disjoint_transfer": len(v17),
            "v18_source_disjoint_transfer": len(v18),
            "v17_population": len(population),
            "v19_source_disjoint_population": len(v19_population),
            "unique_audit_uids": len(
                {row["uid"] for row in [*v15, *v17, *v18]}
            ),
        },
        "v15_metrics": evaluate_rules(v15),
        "v17_metrics": evaluate_rules(v17),
        "v18_metrics": evaluate_rules(v18),
        "v17_inverse_band_weighted_estimates": {
            name: {
                target: weighted_metrics(
                    v17, rule, target, band_counts
                )
                for target in ("visual", "exact", "relabel")
            }
            for name, rule in RULES.items()
        },
        "v17_population_yield": {
            name: sum(bool(rule(row)) for row in population)
            for name, rule in RULES.items()
        },
        "v19_source_disjoint_population_yield": {
            name: sum(bool(rule(row)) for row in v19_population)
            for name, rule in RULES.items()
        },
        "v17_population_score_distribution": dict(
            sorted(Counter(row["visual_score"] for row in population).items())
        ),
        "rule_definitions_sha256": hashlib.sha256(
            "\n".join(RULES).encode()
        ).hexdigest(),
        "artifacts_sha256": {
            "v15_semantic": sha256(
                v15_dir / "audit_selection_semantic.jsonl"
            ),
            "v15_visual": sha256(
                v15_dir / "manual_visual_ledger_blind.tsv"
            ),
            "v15_manual_semantic": sha256(
                v15_dir / "manual_semantic_ledger.tsv"
            ),
            "v17_semantic": sha256(
                v17_dir / "audit_selection_semantic.jsonl"
            ),
            "v17_visual": sha256(
                v17_dir / "manual_visual_ledger_final.tsv"
            ),
            "v17_manual_semantic": sha256(
                v17_dir / "manual_semantic_ledger.tsv"
            ),
            "v18_semantic": sha256(
                v18_dir / "audit_selection_semantic.jsonl"
            ),
            "v18_visual": sha256(
                v18_dir / "manual_visual_ledger_blind.tsv"
            ),
            "v18_manual_semantic": sha256(
                v18_dir / "manual_semantic_ledger.tsv"
            ),
        },
    }
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
