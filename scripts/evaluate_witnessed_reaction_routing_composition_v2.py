#!/usr/bin/env python3
"""Separate witnessed-reaction retrieval from organic-source routing.

The frozen V1 corpus evaluator accidentally used candidate-level staging as
part of its route named ``strict_bystander_reaction``.  That makes a real
bystander response inside a prank or produced scenario look like a reaction
false positive, even though the manual rubric says staging is a later routing
decision.  This additive report preserves the V1 artifact and evaluates four
distinct clip-level decisions:

1. reaction retrieval, ignoring staging;
2. reaction retrieval restricted to social triggers, ignoring staging;
3. organic witnessed routing using the VLM's candidate-level staging answer;
4. organic witnessed routing after audited source-level title/channel cues,
   both with and without the VLM staging answer.

Source cues are exclusion/reroute signals only.  Their absence never certifies
that footage is organic, and none of these routes authorizes corpus mutation or
automatic acceptance.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_witnessed_reaction_atomic_candidates import metric, wilson
    from scripts.evaluate_witnessed_video_asr_corpus_audit import validate_manual
    from scripts.score_witnessed_staging_cues import score_text
    from scripts.select_witnessed_video_asr_corpus_audit import successful_scores
    from scripts.witnessed_reaction_av_contract import candidate_positive
else:
    from evaluate_witnessed_reaction_atomic_candidates import metric, wilson
    from evaluate_witnessed_video_asr_corpus_audit import validate_manual
    from score_witnessed_staging_cues import score_text
    from select_witnessed_video_asr_corpus_audit import successful_scores
    from witnessed_reaction_av_contract import candidate_positive


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


def source_exclusion(provenance: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Return only already-audited strict organic-exclusion mechanisms."""
    rescored = score_text(
        provenance.get("title"), "", provenance.get("channel")
    )
    # Verify that the frozen provenance agrees with the current audited title
    # implementation.  This catches silent regex drift before evaluating.
    expected = {
        "title_staging_cue": rescored["title_staging_cue"],
        "title_creator_initiated_candidate_cue": rescored[
            "title_creator_initiated_candidate_cue"
        ],
    }
    for field, value in expected.items():
        if provenance.get(field) is not value:
            raise ValueError(
                f"{provenance.get('item_id')}: frozen {field} disagrees with "
                "the audited scorer"
            )
    mechanisms = []
    if rescored["witnessed_creator_staging_title_v2"]:
        mechanisms.append("witnessed_creator_staging_title_v2")
    if rescored["witnessed_official_wwyd_channel_v1"]:
        mechanisms.append("witnessed_official_wwyd_channel_v1")
    return bool(mechanisms), tuple(mechanisms)


def base_reaction(result: dict[str, Any], *, require_social: bool) -> bool:
    """Reaction semantics only; staging is deliberately not consulted."""
    return candidate_positive(
        result,
        include_authority=False,
        require_social=require_social,
        require_unstaged=False,
    )


def visual_organic(result: dict[str, Any], *, require_social: bool) -> bool:
    """Reaction semantics plus the candidate-level staging judgment."""
    return candidate_positive(
        result,
        include_authority=False,
        require_social=require_social,
        require_unstaged=True,
    )


def with_prevalence(
    gold: dict[str, bool], predicted: dict[str, bool]
) -> dict[str, Any]:
    report = metric(gold, predicted)
    positives = sum(gold.values())
    report["gold_prevalence"] = positives / len(gold) if gold else None
    report["gold_prevalence_wilson_95"] = wilson(positives, len(gold))
    return report


def evaluate(
    selection_rows: list[dict[str, Any]],
    manual_rows: list[dict[str, str]],
    model_rows: list[dict[str, Any]],
    provenance_rows: list[dict[str, Any]],
    source_review_rows: list[dict[str, str]],
) -> dict[str, Any]:
    validate_manual(manual_rows)

    clip_cohort: dict[str, str] = {}
    clip_uid: dict[str, str] = {}
    candidate_clip: dict[str, str] = {}
    for clip in selection_rows:
        item_id = str(clip.get("item_id") or "")
        uid = str(clip.get("uid") or "")
        if not item_id or not uid or item_id in clip_cohort:
            raise ValueError("selection has missing or duplicate item_id/uid")
        clip_cohort[item_id] = str(clip.get("cohort") or "")
        clip_uid[item_id] = uid
        for candidate in clip.get("candidates") or []:
            candidate_id = str(candidate.get("candidate_id") or "")
            if not candidate_id or candidate_id in candidate_clip:
                raise ValueError("selection has missing or duplicate candidate_id")
            candidate_clip[candidate_id] = item_id
    if len(set(clip_uid.values())) != len(clip_uid):
        raise ValueError("selection is not source-disjoint by uid")

    manual = {str(row.get("candidate_id") or ""): row for row in manual_rows}
    if set(manual) != set(candidate_clip):
        raise ValueError("manual ledger does not exactly cover selection")
    for candidate_id, row in manual.items():
        item_id = candidate_clip[candidate_id]
        if row.get("item_id") != item_id or row.get("uid") != clip_uid[item_id]:
            raise ValueError(f"{candidate_id}: manual lineage mismatch")

    scored = successful_scores(model_rows)
    model: dict[str, dict[str, Any] | None] = {}
    for candidate_id in manual:
        row = scored.get(candidate_id)
        if row is None or row.get("error") or not isinstance(row.get("result"), dict):
            model[candidate_id] = None
            continue
        item_id = candidate_clip[candidate_id]
        if row.get("item_id") != item_id or row.get("uid") != clip_uid[item_id]:
            raise ValueError(f"{candidate_id}: model lineage mismatch")
        model[candidate_id] = row["result"]

    provenance = {str(row.get("item_id") or ""): row for row in provenance_rows}
    if len(provenance) != len(provenance_rows) or set(provenance) != set(clip_cohort):
        raise ValueError("provenance must exactly cover selected clips")
    source_excluded: dict[str, bool] = {}
    source_mechanisms: dict[str, tuple[str, ...]] = {}
    for item_id, row in provenance.items():
        if row.get("uid") != clip_uid[item_id] or row.get("cohort") != clip_cohort[item_id]:
            raise ValueError(f"{item_id}: provenance lineage mismatch")
        excluded, mechanisms = source_exclusion(row)
        source_excluded[item_id] = excluded
        source_mechanisms[item_id] = mechanisms

    triggered_items = {item for item, value in source_excluded.items() if value}
    source_reviews = {
        str(row.get("item_id") or ""): row for row in source_review_rows
    }
    if (
        len(source_reviews) != len(source_review_rows)
        or set(source_reviews) != triggered_items
    ):
        raise ValueError(
            "manual source-cue review must exactly cover triggered selected clips"
        )
    for item_id, row in source_reviews.items():
        if row.get("uid") != clip_uid[item_id] or row.get("cohort") != clip_cohort[item_id]:
            raise ValueError(f"{item_id}: manual source-review lineage mismatch")
        expected_mechanisms = ",".join(source_mechanisms[item_id])
        if row.get("triggered_mechanisms") != expected_mechanisms:
            raise ValueError(f"{item_id}: manual source-review mechanism mismatch")
        if row.get("strict_organic_exclusion_supported") not in {"yes", "no", "uncertain"}:
            raise ValueError(f"{item_id}: invalid manual source-review judgment")
        if row.get("instructional_demo_accepted") not in {"yes", "no", "not_reviewed"}:
            raise ValueError(f"{item_id}: invalid instructional-demo judgment")
        if not row.get("manual_rationale", "").strip():
            raise ValueError(f"{item_id}: missing source-review rationale")

    manual_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    model_by_clip: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate_id, result in manual.items():
        item_id = candidate_clip[candidate_id]
        manual_by_clip[item_id].append(result)
        if model[candidate_id] is not None:
            model_by_clip[item_id].append(model[candidate_id])

    cohort_reports: dict[str, Any] = {}
    for cohort in sorted(set(clip_cohort.values())):
        items = sorted(item for item, value in clip_cohort.items() if value == cohort)
        routes: dict[str, Any] = {}
        for require_social, suffix in ((False, "reaction"), (True, "social_reaction")):
            manual_reaction = {
                item: any(base_reaction(row, require_social=require_social)
                          for row in manual_by_clip[item])
                for item in items
            }
            model_reaction = {
                item: any(base_reaction(row, require_social=require_social)
                          for row in model_by_clip[item])
                for item in items
            }
            manual_visual_organic = {
                item: any(visual_organic(row, require_social=require_social)
                          for row in manual_by_clip[item])
                for item in items
            }
            model_visual_organic = {
                item: any(visual_organic(row, require_social=require_social)
                          for row in model_by_clip[item])
                for item in items
            }
            # Post-reveal target: a reaction clip is organic only if neither
            # the blinded visual audit nor an audited source cue establishes
            # staging.  Uncertain visual staging already fails closed.
            post_reveal_gold = {
                item: manual_visual_organic[item] and not source_excluded[item]
                for item in items
            }
            source_only_prediction = {
                item: model_reaction[item] and not source_excluded[item]
                for item in items
            }
            source_plus_visual_prediction = {
                item: model_visual_organic[item] and not source_excluded[item]
                for item in items
            }
            routes[f"strict_bystander_{suffix}_staging_independent"] = (
                with_prevalence(manual_reaction, model_reaction)
            )
            routes[f"organic_{suffix}_vlm_staging_only"] = (
                with_prevalence(manual_visual_organic, model_visual_organic)
            )
            routes[f"organic_{suffix}_audited_source_cues_only"] = (
                with_prevalence(post_reveal_gold, source_only_prediction)
            )
            routes[f"organic_{suffix}_vlm_plus_audited_source_cues"] = (
                with_prevalence(post_reveal_gold, source_plus_visual_prediction)
            )

        excluded_items = [item for item in items if source_excluded[item]]
        manual_reaction_items = {
            item for item in items
            if any(base_reaction(row, require_social=False)
                   for row in manual_by_clip[item])
        }
        model_reaction_items = {
            item for item in items
            if any(base_reaction(row, require_social=False)
                   for row in model_by_clip[item])
        }
        blind_organic_items = {
            item for item in items
            if any(visual_organic(row, require_social=False)
                   for row in manual_by_clip[item])
        }
        cohort_reports[cohort] = {
            "clips": len(items),
            "routes": routes,
            "audited_source_routing": {
                "source_excluded_clips": len(excluded_items),
                "source_excluded_item_ids": excluded_items,
                "manual_reaction_positive_reroutes": sorted(
                    manual_reaction_items & set(excluded_items)
                ),
                "model_reaction_positive_reroutes": sorted(
                    model_reaction_items & set(excluded_items)
                ),
                "blind_manual_organic_labels_overridden_by_source_evidence": sorted(
                    blind_organic_items & set(excluded_items)
                ),
                "mechanism_counts": dict(sorted(Counter(
                    mechanism
                    for item in excluded_items
                    for mechanism in source_mechanisms[item]
                ).items())),
            },
            "interpretation": (
                "unbiased_one_representative_clip_per_source_estimate_not_clip_weighted"
                if cohort == "uniform_probability_sample"
                else "enriched_error_discovery_not_a_corpus_estimate"
            ),
        }

    return {
        "kind": "witnessed_reaction_routing_composition_audit_v2",
        "policy": "manual_audit_shadow_routing_only_no_automatic_acceptance",
        "semantic_correction": (
            "reaction retrieval ignores staging; organic-source eligibility is "
            "reported as a separate downstream route"
        ),
        "selected_clips": len(clip_cohort),
        "selected_candidates": len(candidate_clip),
        "source_disjoint": len(set(clip_uid.values())) == len(clip_uid),
        "manual_candidate_review_complete": len(manual) == len(candidate_clip),
        "model_candidate_coverage": (
            sum(row is not None for row in model.values()) / len(model) if model else None
        ),
        "audited_source_cue_coverage": len(provenance) / len(clip_cohort),
        "source_cue_transfer_manual_review_complete": (
            set(source_reviews) == triggered_items
        ),
        "source_cue_transfer_manual_review": {
            "triggered": len(source_reviews),
            "supported": sum(
                row["strict_organic_exclusion_supported"] == "yes"
                for row in source_reviews.values()
            ),
            "unsupported": sum(
                row["strict_organic_exclusion_supported"] == "no"
                for row in source_reviews.values()
            ),
            "uncertain": sum(
                row["strict_organic_exclusion_supported"] == "uncertain"
                for row in source_reviews.values()
            ),
            "instructional_demo_accepted": sum(
                row["instructional_demo_accepted"] == "yes"
                for row in source_reviews.values()
            ),
        },
        "source_cue_absence_certifies_organic": False,
        "headline_source_uniform_estimate_cohort": "uniform_probability_sample",
        "clip_cohort_metrics": cohort_reports,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        read_jsonl(args.selection),
        read_tsv(args.manual),
        read_jsonl(args.model),
        read_jsonl(args.provenance),
        read_tsv(args.source_review),
    )
    report["artifacts_sha256"] = {
        "selection": sha256(args.selection),
        "manual": sha256(args.manual),
        "model": sha256(args.model),
        "provenance": sha256(args.provenance),
        "source_review": sha256(args.source_review),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
