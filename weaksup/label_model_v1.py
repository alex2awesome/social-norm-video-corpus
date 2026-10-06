#!/usr/bin/env python3
"""Pillar/target shadow label model over standardized LF records.

An auditable Snorkel-style generative combiner (roadmap sections 12.4-12.8):
votes are first collapsed to one effective vote per evidence family (correlated
LFs cannot multiply), an EM-fit naive-Bayes-with-abstention model estimates
per-family accuracies and a class prior without gold labels, and posteriors are
banded into append-only shadow routes.  Deterministic eligibility gates are
never outvoted.  Baselines (family majority vote, strongest single LF chosen on
the train split) are reported alongside so the model must earn its keep.
Outputs are shadow probabilities only: no acceptance label, no deletion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

try:
    from weaksup.labeling_functions_v1 import eligibility_gate
    from weaksup.lf_matrix_v1 import build_matrices, iter_jsonl, wilson_lower
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from weaksup.labeling_functions_v1 import eligibility_gate
    from weaksup.lf_matrix_v1 import build_matrices, iter_jsonl, wilson_lower


MODEL_VERSION = "label_model_v1"

DEFAULT_HIGH_BAND = 0.85
DEFAULT_LOW_BAND = 0.15
# Accuracies are constrained to better-than-chance: EM may learn HOW reliable
# a family is, but may never polarity-flip it.  Without this floor a dominant
# correlated positive block makes EM invert audited negative cues (observed on
# the 2026-08-17 v2 corpus fit: the precision-1.0 staging cue was assigned
# accuracy 0.05 and its votes counted as positive evidence).
ACCURACY_FLOOR, ACCURACY_CEIL = 0.55, 0.95
PRIOR_FLOOR, PRIOR_CEIL = 0.01, 0.99

BANDS = (
    "high_confidence_candidate",
    "disagreement_manual_review",
    "likely_failure_or_reroute",
    "insufficient_evidence_abstain",
    "gate_failed_review",
)


def collapse_families(
    matrix: dict[str, Any], item: str
) -> dict[str, int]:
    """One effective vote per evidence family: any -1 wins, else +1 if any +1,
    else abstain.  Within-family conflict is resolved fail-closed."""
    by_family: dict[str, list[int]] = {}
    for lf_id, family in matrix["lfs"].items():
        vote = matrix["votes"].get((item, lf_id), 0)
        if vote != 0:
            by_family.setdefault(family, []).append(vote)
    collapsed = {}
    for family, votes in by_family.items():
        collapsed[family] = -1 if -1 in votes else 1
    return collapsed


def family_vote_matrix(matrix: dict[str, Any]) -> dict[str, dict[str, int]]:
    return {item: collapse_families(matrix, item) for item in sorted(matrix["items"])}


def majority_vote(family_votes: dict[str, int]) -> int:
    total = sum(family_votes.values())
    return 1 if total > 0 else -1 if total < 0 else 0


def source_disjoint_split(
    items: list[str],
    group_key: dict[str, str],
    *,
    train_fraction: float = 0.6,
    calibration_fraction: float = 0.2,
    salt: str = "label_model_v1_split",
) -> dict[str, str]:
    """Deterministic hash-based split grouped by source/cluster key.  Every
    item in one group lands in one split, so sources never straddle splits."""
    assignments = {}
    for item in items:
        group = group_key.get(item, item)
        digest = hashlib.sha256(f"{salt}:{group}".encode()).hexdigest()
        fraction = int(digest[:12], 16) / 16**12
        if fraction < train_fraction:
            assignments[item] = "train"
        elif fraction < train_fraction + calibration_fraction:
            assignments[item] = "calibration"
        else:
            assignments[item] = "test"
    return assignments


class FamilyLabelModel:
    """Naive-Bayes generative model over family votes with abstention.

    Abstentions are treated as missing evidence (no likelihood term), matching
    the project's rule that missing evidence yields abstention, not certainty.
    """

    def __init__(self, families: list[str]):
        self.families = sorted(families)
        self.prior_positive = 0.5
        self.accuracy = {family: 0.7 for family in self.families}
        self.fitted_iterations = 0

    def posterior_positive(self, family_votes: dict[str, int]) -> float:
        log_pos = math.log(self.prior_positive)
        log_neg = math.log(1 - self.prior_positive)
        for family in self.families:
            vote = family_votes.get(family, 0)
            if vote == 0:
                continue
            acc = self.accuracy[family]
            log_pos += math.log(acc if vote == 1 else 1 - acc)
            log_neg += math.log(acc if vote == -1 else 1 - acc)
        peak = max(log_pos, log_neg)
        pos, neg = math.exp(log_pos - peak), math.exp(log_neg - peak)
        return pos / (pos + neg)

    def fit(
        self,
        vote_rows: list[dict[str, int]],
        *,
        max_iterations: int = 200,
        tolerance: float = 1e-6,
    ) -> "FamilyLabelModel":
        if not vote_rows:
            raise ValueError("cannot fit on zero rows")
        for _ in range(max_iterations):
            posteriors = [self.posterior_positive(row) for row in vote_rows]
            new_prior = min(
                PRIOR_CEIL, max(PRIOR_FLOOR, sum(posteriors) / len(posteriors))
            )
            new_accuracy = {}
            for family in self.families:
                correct = total = 0.0
                for row, p in zip(vote_rows, posteriors):
                    vote = row.get(family, 0)
                    if vote == 0:
                        continue
                    total += 1.0
                    correct += p if vote == 1 else (1 - p)
                if total:
                    new_accuracy[family] = min(
                        ACCURACY_CEIL, max(ACCURACY_FLOOR, correct / total)
                    )
                else:
                    new_accuracy[family] = self.accuracy[family]
            delta = abs(new_prior - self.prior_positive) + sum(
                abs(new_accuracy[f] - self.accuracy[f]) for f in self.families
            )
            self.prior_positive, self.accuracy = new_prior, new_accuracy
            self.fitted_iterations += 1
            if delta < tolerance:
                break
        return self

    def parameters(self) -> dict[str, Any]:
        return {
            "model_version": MODEL_VERSION,
            "prior_positive": self.prior_positive,
            "family_accuracy": dict(self.accuracy),
            "fitted_iterations": self.fitted_iterations,
        }


def calibration_report(
    probabilities: list[float], labels: list[int], *, bins: int = 10
) -> dict[str, Any]:
    """Reliability bins and expected calibration error on audited labels."""
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must align")
    rows = []
    total = len(probabilities)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        members = [
            (p, label)
            for p, label in zip(probabilities, labels)
            if lo <= p < hi or (b == bins - 1 and p == 1.0)
        ]
        if not members:
            continue
        mean_p = sum(p for p, _ in members) / len(members)
        positives = sum(label == 1 for _, label in members)
        rate = positives / len(members)
        ece += len(members) / total * abs(mean_p - rate) if total else 0.0
        rows.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "count": len(members),
                "mean_probability": mean_p,
                "empirical_positive_rate": rate,
                "wilson_lower": wilson_lower(positives, len(members)),
            }
        )
    return {"bins": rows, "expected_calibration_error": ece, "n": total}


def evaluate_predictions(
    predictions: dict[str, int], gold: dict[str, int]
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for item, label in gold.items():
        pred = predictions.get(item, 0)
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == -1:
            fp += 1
        elif label == 1:
            fn += 1
        else:
            tn += 1
    selected = tp + fp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": tp / selected if selected else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "precision_wilson_lower": wilson_lower(tp, selected),
    }


def strongest_single_lf(
    matrix: dict[str, Any], gold: dict[str, int], train_items: set[str]
) -> str | None:
    """Pick the best single LF by F1 on the train split only, so the baseline
    never peeks at calibration or test labels."""
    best_lf, best_f1 = None, -1.0
    for lf_id in sorted(matrix["lfs"]):
        predictions = {
            item: matrix["votes"].get((item, lf_id), 0) for item in train_items
        }
        train_gold = {k: v for k, v in gold.items() if k in train_items}
        stats = evaluate_predictions(predictions, train_gold)
        precision, recall = stats["precision"], stats["recall"]
        if precision is None or recall is None or precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_lf, best_f1 = lf_id, f1
    return best_lf


def shadow_band(
    posterior: float,
    family_votes: dict[str, int],
    gate: dict[str, Any],
    *,
    high: float = DEFAULT_HIGH_BAND,
    low: float = DEFAULT_LOW_BAND,
) -> str:
    """Append-only shadow banding (roadmap section 12.8).  Gates dominate:
    twenty transcript votes cannot compensate for a failed eligibility gate."""
    if gate["failed_gates"]:
        return "gate_failed_review"
    has_conflict = 1 in family_votes.values() and -1 in family_votes.values()
    if not family_votes or gate["unknown_gates"]:
        return "insufficient_evidence_abstain"
    if has_conflict:
        return "disagreement_manual_review"
    # A saturated fitted prior can push vote-free-of-support items past the
    # threshold; high confidence additionally requires actual positive
    # evidence, and prior-carried items without any negative stay abstained.
    if posterior >= high:
        return (
            "high_confidence_candidate"
            if 1 in family_votes.values()
            else "insufficient_evidence_abstain"
        )
    if posterior <= low:
        return "likely_failure_or_reroute"
    return "insufficient_evidence_abstain"


def score_matrix(
    matrix: dict[str, Any],
    *,
    pillar: str,
    target: str,
    gates: dict[str, dict[str, Any]] | None = None,
    high: float = DEFAULT_HIGH_BAND,
    low: float = DEFAULT_LOW_BAND,
) -> tuple[FamilyLabelModel, list[dict[str, Any]]]:
    """Fit the family label model on all items and emit shadow rows."""
    votes_by_item = family_vote_matrix(matrix)
    families = sorted(set(matrix["lfs"].values()))
    model = FamilyLabelModel(families).fit(list(votes_by_item.values()))
    rows = []
    for item, family_votes in votes_by_item.items():
        gate = (gates or {}).get(
            item, {"eligible": None, "failed_gates": [], "unknown_gates": []}
        )
        posterior = model.posterior_positive(family_votes)
        rows.append(
            {
                "item_id": item,
                "pillar": pillar,
                "target": target,
                "model_version": MODEL_VERSION,
                "posterior_positive": posterior,
                "family_votes": dict(sorted(family_votes.items())),
                "majority_vote": majority_vote(family_votes),
                "gate": gate,
                "shadow_band": shadow_band(
                    posterior, family_votes, gate, high=high, low=low
                ),
                "acceptance_label": None,
                "corpus_disposition": None,
                "delete_media": False,
                "shadow_only": True,
            }
        )
    return model, rows


def run_experiment(
    matrix: dict[str, Any],
    gold: dict[str, int],
    group_key: dict[str, str],
    *,
    pillar: str,
    target: str,
    high: float = DEFAULT_HIGH_BAND,
) -> dict[str, Any]:
    """Full section-12.7 procedure on one (pillar, target) matrix: source-
    disjoint split, fit on train, calibrate on calibration, evaluate on the
    untouched test split against majority-vote and strongest-LF baselines."""
    items = sorted(matrix["items"])
    split = source_disjoint_split(items, group_key)
    train = [item for item in items if split[item] == "train"]
    calibration = [item for item in items if split[item] == "calibration"]
    test = [item for item in items if split[item] == "test"]

    votes_by_item = family_vote_matrix(matrix)
    families = sorted(set(matrix["lfs"].values()))
    model = FamilyLabelModel(families).fit([votes_by_item[item] for item in train])

    def predictions_for(item_ids: list[str], threshold: float) -> dict[str, int]:
        result = {}
        for item in item_ids:
            p = model.posterior_positive(votes_by_item[item])
            result[item] = 1 if p >= threshold else -1 if p <= 1 - threshold else 0
        return result

    calibration_gold = [
        (model.posterior_positive(votes_by_item[item]), gold[item])
        for item in calibration
        if item in gold
    ]
    test_gold = {item: gold[item] for item in test if item in gold}
    best_lf = strongest_single_lf(matrix, gold, set(train))
    baselines = {
        "majority_vote": evaluate_predictions(
            {item: majority_vote(votes_by_item[item]) for item in test}, test_gold
        ),
        "strongest_single_lf": {
            "lf_id": best_lf,
            **evaluate_predictions(
                {
                    item: matrix["votes"].get((item, best_lf), 0) if best_lf else 0
                    for item in test
                },
                test_gold,
            ),
        },
    }
    return {
        "pillar": pillar,
        "target": target,
        "split_sizes": {
            "train": len(train),
            "calibration": len(calibration),
            "test": len(test),
        },
        "model": model.parameters(),
        "calibration": calibration_report(
            [p for p, _ in calibration_gold], [label for _, label in calibration_gold]
        )
        if calibration_gold
        else None,
        "label_model_test": evaluate_predictions(
            predictions_for(test, high), test_gold
        ),
        "baselines": baselines,
        "test_gold_items": len(test_gold),
        "policy": "shadow_only_no_acceptance_no_deletion",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--gates", type=Path, default=None,
                        help="JSONL of per-item gate fields (item_id + gate columns)")
    parser.add_argument("--pillar", required=True,
                        choices=("witnessed", "instructional", "commentary"))
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--high", type=float, default=DEFAULT_HIGH_BAND)
    parser.add_argument("--low", type=float, default=DEFAULT_LOW_BAND)
    args = parser.parse_args()
    for path in (args.out, args.summary_out):
        if path.exists():
            raise FileExistsError(f"output exists: {path}")

    matrices = build_matrices(iter_jsonl(args.records))
    key = (args.pillar, args.target)
    if key not in matrices:
        raise ValueError(f"no records for {key}")
    gates = {}
    if args.gates:
        for row in iter_jsonl(args.gates):
            gates[row["item_id"]] = eligibility_gate(row, args.pillar)
    model, rows = score_matrix(
        matrices[key], pillar=args.pillar, target=args.target,
        gates=gates, high=args.high, low=args.low,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    band_counts: dict[str, int] = {}
    for row in rows:
        band_counts[row["shadow_band"]] = band_counts.get(row["shadow_band"], 0) + 1
    summary = {
        "rows": len(rows),
        "bands": band_counts,
        "model": model.parameters(),
        "acceptance_rows": 0,
        "deleted_rows": 0,
        "input_mutated": False,
    }
    args.summary_out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
