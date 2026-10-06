#!/usr/bin/env python3
"""Evaluate witnessed-reaction models against candidate-level atomic judgments.

The frozen manual ledger predates this evaluator.  It deliberately separates
reaction presence, responder identity, social-trigger semantics, and routing.
Unresolved manual judgments are excluded from the corresponding atom instead
of silently becoming negatives.  Model errors and abstentions fail closed for
composed routing rules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Callable


SOCIAL_TRIGGERS = {"interpersonal_treatment", "shared_public_conduct"}
TARGETED_RESPONSES = {
    "targeted_objection",
    "correction_or_sanction",
    "protective_intervention",
    "interposition_or_separation",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


def manual_atoms(row: dict[str, str]) -> dict[str, bool | None]:
    response = row["visual_third_party_response"]
    role = row["visual_identity_role"]
    disposition = row["speaker_proxy_disposition"]
    response_present = {"yes": True, "no": False}.get(response)
    role_resolved = role not in {"bystander_uncertain", "authority_offscreen"}
    bystander_identity = (
        role in {"bystander", "generic_safety_bystander"}
        if role_resolved
        else None
    )
    # Trigger semantics were manually resolved only where a third-party visual
    # response exists.  Authority cases remain a distinct route here.
    social_trigger: bool | None = None
    if response == "yes" and role in {"bystander", "generic_safety_bystander"}:
        social_trigger = disposition != "failed_social_trigger"

    witnessed_review: bool | None = None
    instructional_reroute: bool | None = None
    if response != "uncertain":
        witnessed_review = (
            response == "yes"
            and role == "bystander"
            and disposition == "candidate_generation_only"
        )
        instructional_reroute = (
            response == "yes" and disposition == "instructional_reroute_only"
        )
    return {
        "response_present": response_present,
        "bystander_identity": bystander_identity,
        "social_trigger": social_trigger,
        "witnessed_review": witnessed_review,
        "instructional_reroute": instructional_reroute,
    }


def model_atoms(row: dict[str, Any]) -> dict[str, bool | None]:
    if row.get("error") or not isinstance(row.get("result"), dict):
        return {name: None for name in (
            "response_present", "bystander_identity", "social_trigger",
            "witnessed_review", "instructional_reroute",
        )}
    result = row["result"]
    if "reaction_grounded" in result:  # audiovisual V3 schema
        response_present = (
            result.get("reaction_grounded") == "yes"
            and result.get("response_targets_action") == "yes"
        )
        bystander_identity = result.get("responder_role") == "separate_bystander"
        social_trigger = result.get("trigger_kind") in SOCIAL_TRIGGERS
        targeted = result.get("response_content") in TARGETED_RESPONSES
        base = response_present and bystander_identity and social_trigger and targeted
        return {
            "response_present": response_present,
            "bystander_identity": bystander_identity,
            "social_trigger": social_trigger,
            "witnessed_review": base and result.get("staging") != "clearly_staged",
            "instructional_reroute": base and result.get("staging") == "clearly_staged",
        }
    if "visible_response_at_candidate_moment" in result:  # visual-only V2
        response_present = (
            result.get("visible_response_at_candidate_moment") == "yes"
            and result.get("visible_response_targets_action") == "yes"
        )
        return {
            "response_present": response_present,
            "bystander_identity": result.get("visual_responder_role")
            in {"bystander", "organic_audience"},
            "social_trigger": None,
            "witnessed_review": None,
            "instructional_reroute": None,
        }
    raise ValueError("unrecognized witnessed candidate model schema")


def metric(gold: dict[str, bool], predicted: dict[str, bool | None]) -> dict[str, Any]:
    ids = sorted(gold)
    observed = [item for item in ids if predicted.get(item) is not None]

    def counts(items: list[str], fail_closed: bool) -> dict[str, Any]:
        values = {
            item: bool(predicted.get(item)) if fail_closed else bool(predicted[item])
            for item in items
        }
        tp = sum(gold[item] and values[item] for item in items)
        fp = sum(not gold[item] and values[item] for item in items)
        fn = sum(gold[item] and not values[item] for item in items)
        tn = sum(not gold[item] and not values[item] for item in items)
        tp_ids = [item for item in items if gold[item] and values[item]]
        fp_ids = [item for item in items if not gold[item] and values[item]]
        fn_ids = [item for item in items if gold[item] and not values[item]]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        return {
            "items": len(items), "selected": tp + fp, "tp": tp, "fp": fp,
            "fn": fn, "tn": tn, "precision": precision, "recall": recall,
            "precision_wilson_95": wilson(tp, tp + fp),
            "recall_wilson_95": wilson(tp, tp + fn),
            "tp_candidate_ids": tp_ids,
            "fp_candidate_ids": fp_ids,
            "fn_candidate_ids": fn_ids,
        }

    return {
        "gold_items": len(ids),
        "gold_positives": sum(gold.values()),
        "model_observed": len(observed),
        "coverage": len(observed) / len(ids) if ids else None,
        "observed_only": counts(observed, False),
        "fail_closed": counts(ids, True),
    }


def evaluate(
    manual_rows: list[dict[str, str]],
    model_sets: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    manual = {row["candidate_id"]: row for row in manual_rows}
    if len(manual) != len(manual_rows):
        raise ValueError("manual ledger contains duplicate candidate_id")
    manual_by_atom = {
        atom: {
            item: bool(value)
            for item, row in manual.items()
            if (value := manual_atoms(row)[atom]) is not None
        }
        for atom in manual_atoms(manual_rows[0])
    }
    reports: dict[str, Any] = {}
    predictions: dict[str, dict[str, dict[str, bool | None]]] = {}
    for model, rows in model_sets.items():
        keyed = {str(row["candidate_id"]): row for row in rows}
        if len(keyed) != len(rows):
            raise ValueError(f"{model} contains duplicate candidate_id")
        if set(keyed) != set(manual):
            raise ValueError(f"{model} candidate coverage differs from manual ledger")
        predictions[model] = {
            atom: {item: model_atoms(keyed[item])[atom] for item in manual}
            for atom in manual_by_atom
        }
        reports[model] = {
            atom: metric(gold, predictions[model][atom])
            for atom, gold in manual_by_atom.items()
        }
    if len(predictions) >= 2:
        # Strict consensus: any abstention or disagreement fails closed.
        consensus = {
            atom: {
                item: all(
                    prediction[atom][item] is True
                    for prediction in predictions.values()
                )
                for item in manual
            }
            for atom in manual_by_atom
        }
        reports["strict_model_consensus"] = {
            atom: metric(gold, consensus[atom])
            for atom, gold in manual_by_atom.items()
        }
    return {
        "kind": "witnessed_reaction_atomic_candidate_audit_v1",
        "policy": "development_audit_only_no_acceptance_or_corpus_mutation",
        "manual_candidates": len(manual),
        "manual_gold_counts": {
            atom: {"items": len(gold), "positive": sum(gold.values())}
            for atom, gold in manual_by_atom.items()
        },
        "models": reports,
    }


def disagreement_rows(
    manual_rows: list[dict[str, str]],
    model_sets: dict[str, list[dict[str, Any]]],
) -> list[dict[str, str]]:
    manual = {row["candidate_id"]: row for row in manual_rows}
    output = []
    for model, rows in model_sets.items():
        keyed = {str(row["candidate_id"]): row for row in rows}
        for candidate_id in sorted(manual, key=lambda item: int(manual[item]["storyboard_index"])):
            gold = manual_atoms(manual[candidate_id])
            predicted = model_atoms(keyed[candidate_id])
            mismatches = [
                atom for atom, value in gold.items()
                if value is not None and predicted[atom] != value
            ]
            if not mismatches:
                continue
            result = keyed[candidate_id].get("result") or {}
            output.append({
                "model": model,
                "storyboard_index": manual[candidate_id]["storyboard_index"],
                "candidate_id": candidate_id,
                "mismatch_atoms": ",".join(mismatches),
                "manual_identity_role": manual[candidate_id]["visual_identity_role"],
                "manual_response": manual[candidate_id]["visual_third_party_response"],
                "manual_disposition": manual[candidate_id]["speaker_proxy_disposition"],
                "predicted_responder_role": str(result.get("responder_role") or result.get("visual_responder_role") or ""),
                "predicted_trigger_kind": str(result.get("trigger_kind") or "not_asked"),
                "predicted_staging": str(result.get("staging") or "not_asked"),
                "model_evidence": str(result.get("evidence") or ""),
                "manual_note": manual[candidate_id].get("manual_note", ""),
            })
    return output


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["model", "candidate_id", "mismatch_atoms"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument(
        "--model-output", action="append", nargs=2, metavar=("NAME", "PATH"),
        required=True,
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--disagreements-out", type=Path)
    args = parser.parse_args()
    manual_rows = read_tsv(args.manual)
    model_sets = {name: read_jsonl(Path(path)) for name, path in args.model_output}
    report = evaluate(
        manual_rows,
        model_sets,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.disagreements_out:
        write_tsv(args.disagreements_out, disagreement_rows(manual_rows, model_sets))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
