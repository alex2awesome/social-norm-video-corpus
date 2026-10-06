#!/usr/bin/env python3
"""Evaluate the preregistered instructional visual/title transfer audit.

The rule remains non-destructive.  Passing permits only candidate generation
and manual-review ranking; it never authorizes acceptance, rejection, deletion,
or corpus mutation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from math import sqrt
from pathlib import Path
from typing import Any, Callable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yes(value: Any) -> bool:
    return str(value).strip().casefold() == "yes"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if not n:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = sqrt(p * (1 - p) / n + z * z / (4 * n * n)) * z / denominator
    return [center - half, center + half]


def metric(
    rows: list[dict[str, Any]], selector: Callable[[dict[str, Any]], bool]
) -> dict[str, Any]:
    selected = [row for row in rows if selector(row)]
    positives = [row for row in rows if row["manual_visual_demo"]]
    tp = sum(row["manual_visual_demo"] for row in selected)
    fp = len(selected) - tp
    fn = len(positives) - tp
    return {
        "n": len(rows),
        "manual_positives": len(positives),
        "selected": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / len(selected) if selected else None,
        "precision_wilson_95": wilson(tp, len(selected)),
        "recall": tp / len(positives) if positives else None,
    }


def unique_by(
    rows: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any], label: str
) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for row in rows:
        row_key = key(row)
        if row_key in result:
            raise ValueError(f"duplicate {label}: {row_key!r}")
        result[row_key] = row
    return result


def evaluate(
    preregistration: dict[str, Any],
    sealed_rows: list[dict[str, Any]],
    storyboard_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
    gemma_rows: list[dict[str, Any]],
    blind_rows: list[dict[str, str]],
    semantic_rows: list[dict[str, str]],
    output_audit_rows: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sealed = unique_by(sealed_rows, lambda row: int(row["audit_index"]), "audit_index")
    if set(sealed) != set(range(int(preregistration["items"]))):
        raise ValueError("sealed selection must cover every preregistered audit index")
    if len({str(row["uid"]) for row in sealed.values()}) != len(sealed):
        raise ValueError("transfer cohort is not source-disjoint")

    item_to_index = unique_by(
        sealed_rows, lambda row: str(row["item_id"]), "sealed item_id"
    )
    item_index = {item_id: int(row["audit_index"]) for item_id, row in item_to_index.items()}
    storyboards = unique_by(
        storyboard_rows, lambda row: str(row["item_id"]), "storyboard item_id"
    )
    blind = unique_by(blind_rows, lambda row: int(row["audit_index"]), "blind audit_index")
    semantic = unique_by(
        semantic_rows, lambda row: int(row["audit_index"]), "semantic audit_index"
    )
    if set(blind) != set(sealed) or set(semantic) != set(sealed):
        raise ValueError("manual ledgers must exactly cover the sealed cohort")

    model_rows: dict[tuple[int, str], dict[str, Any]] = {}
    for row in qwen_rows + gemma_rows:
        item_id = str(row["item_id"])
        if item_id not in item_index:
            raise ValueError(f"model output has unknown item_id: {item_id}")
        key = (item_index[item_id], str(row["model"]))
        if key in model_rows:
            raise ValueError(f"duplicate model output: {key!r}")
        if row.get("error") is not None:
            raise ValueError(f"model output contains an error: {key!r}")
        model_rows[key] = row

    audit = unique_by(
        output_audit_rows,
        lambda row: (int(row["audit_index"]), str(row["model"])),
        "manual model-output audit key",
    )
    expected_models = {str(row["model"]) for row in qwen_rows + gemma_rows}
    if len(expected_models) != 2:
        raise ValueError("expected exactly two frozen VLMs")
    rendered_indices = {
        int(row["audit_index"])
        for row in blind_rows
        if row["render_status"] == "success"
    }
    expected_output_keys = {
        (index, model) for index in rendered_indices for model in expected_models
    }
    if set(model_rows) != expected_output_keys:
        raise ValueError("model outputs do not exactly cover every rendered item")
    if set(audit) != expected_output_keys:
        raise ValueError("manual output audit does not exactly cover model outputs")
    if set(storyboards) != set(item_index):
        raise ValueError("storyboard manifest does not exactly cover the sealed cohort")
    successful_storyboards = {
        item_id for item_id, row in storyboards.items()
        if not row.get("error") and int(row.get("frame_count") or 0) > 0
    }
    expected_rendered_items = {
        str(sealed[index]["item_id"]) for index in rendered_indices
    }
    if successful_storyboards != expected_rendered_items:
        raise ValueError("successful storyboard rows do not match the blind render ledger")

    qwen_model = next(model for model in expected_models if model.startswith("qwen"))
    gemma_model = next(model for model in expected_models if model.startswith("gemma"))
    normalized: list[dict[str, Any]] = []
    for index in sorted(rendered_indices):
        manual = blind[index]
        if manual["manual_visual_demo"] not in {"yes", "no"}:
            raise ValueError(f"rendered item {index} lacks binary manual gold")
        qwen = model_rows[(index, qwen_model)]
        gemma = model_rows[(index, gemma_model)]
        qwen_pass = yes(qwen["result"]["demo_pass"])
        gemma_pass = yes(gemma["result"]["demo_pass"])
        title_cue = bool(sealed[index]["scene_title_candidate"])
        normalized.append({
            "audit_index": index,
            "item_id": str(sealed[index]["item_id"]),
            "uid": str(sealed[index]["uid"]),
            "scene_title_candidate": title_cue,
            "manual_visual_demo": yes(manual["manual_visual_demo"]),
            "qwen_demo_pass": qwen_pass,
            "gemma_demo_pass": gemma_pass,
            "dual_vlm_demo_pass": qwen_pass and gemma_pass,
            "visual_title_conjunction": qwen_pass and gemma_pass and title_cue,
            "original_label_usable": yes(semantic[index]["original_label_usable"]),
            "label_alignment": semantic[index]["label_alignment"],
            "scene_grounded_relabel_needed": yes(
                semantic[index]["scene_grounded_relabel_needed"]
            ),
            "qwen_output_visually_supported": yes(
                audit[(index, qwen_model)]["output_visually_supported"]
            ),
            "gemma_output_visually_supported": yes(
                audit[(index, gemma_model)]["output_visually_supported"]
            ),
        })

    conjunction = metric(normalized, lambda row: row["visual_title_conjunction"])
    gate = preregistration["gate"]
    lower = conjunction["precision_wilson_95"][0] if conjunction["precision_wilson_95"] else None
    checks = {
        "minimum_rendered_items": len(rendered_indices) >= int(gate["minimum_rendered_items"]),
        "manual_storyboard_coverage": len(blind) == int(preregistration["items"]),
        "manual_semantic_coverage": len(semantic) == int(preregistration["items"]),
        "manual_model_output_coverage": len(audit) == len(expected_output_keys),
        "minimum_selected": conjunction["selected"] >= int(gate["minimum_selected"]),
        "minimum_visual_demo_precision": (
            conjunction["precision"] is not None
            and conjunction["precision"] >= float(gate["minimum_visual_demo_precision"])
        ),
        "minimum_visual_demo_precision_wilson_95_lower": (
            lower is not None
            and lower >= float(gate["minimum_visual_demo_precision_wilson_95_lower"])
        ),
    }
    passed = all(checks.values())

    selected = [row for row in normalized if row["visual_title_conjunction"]]
    by_model: dict[str, dict[str, int]] = {}
    for model in sorted(expected_models):
        model_audits = [row for key, row in audit.items() if key[1] == model]
        by_model[model] = {
            "reviewed": len(model_audits),
            "visually_supported": sum(yes(row["output_visually_supported"]) for row in model_audits),
            "visually_unsupported": sum(not yes(row["output_visually_supported"]) for row in model_audits),
        }
    error_counts = Counter(row["error_mechanism"] for row in output_audit_rows)

    selectors = {
        "qwen": lambda row: row["qwen_demo_pass"],
        "gemma": lambda row: row["gemma_demo_pass"],
        "dual_vlm": lambda row: row["dual_vlm_demo_pass"],
        "visual_title_conjunction": lambda row: row["visual_title_conjunction"],
    }
    title_positive = [row for row in normalized if row["scene_title_candidate"]]
    title_negative = [row for row in normalized if not row["scene_title_candidate"]]
    report = {
        "kind": preregistration["kind"],
        "items": int(preregistration["items"]),
        "rendered_items": len(rendered_indices),
        "decode_failures": int(preregistration["items"]) - len(rendered_indices),
        "manual_visual_demos": sum(row["manual_visual_demo"] for row in normalized),
        "manual_review_complete": True,
        "manual_model_outputs_reviewed": len(audit),
        "metrics": {name: metric(normalized, selector) for name, selector in selectors.items()},
        "title_strata": {
            "title_positive": {
                name: metric(title_positive, selector) for name, selector in selectors.items()
            },
            "title_negative_control": {
                name: metric(title_negative, selector) for name, selector in selectors.items()
            },
        },
        "selected_semantic_quality": {
            "selected": len(selected),
            "original_label_usable": sum(row["original_label_usable"] for row in selected),
            "exact_label_alignment": sum(row["label_alignment"] == "exact" for row in selected),
            "scene_grounded_relabel_needed": sum(
                row["scene_grounded_relabel_needed"] for row in selected
            ),
        },
        "manual_output_audit": {
            "by_model": by_model,
            "error_mechanisms": dict(sorted(error_counts.items())),
        },
        "checks": checks,
        "preregistered_pass": passed,
        "status": (
            "passed_for_candidate_generation_and_manual_review_ranking_only"
            if passed else "failed_transfer"
        ),
        "allowed_uses": (
            ["candidate_generation", "manual_review_ranking"] if passed else []
        ),
        "automatic_acceptance": False,
        "automatic_rejection": False,
        "corpus_mutation_authorized": False,
        "all_media_retained": True,
    }
    return normalized, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--storyboards", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--gemma", type=Path, required=True)
    parser.add_argument("--manual-blind", type=Path, required=True)
    parser.add_argument("--manual-semantic", type=Path, required=True)
    parser.add_argument("--manual-output-audit", type=Path, required=True)
    parser.add_argument("--rows-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    args = parser.parse_args()
    if args.rows_out.exists() or args.summary_out.exists():
        raise SystemExit("refusing to overwrite transfer artifacts")
    inputs = {
        "preregistration": args.preregistration,
        "sealed": args.sealed,
        "storyboards": args.storyboards,
        "qwen": args.qwen,
        "gemma": args.gemma,
        "manual_blind": args.manual_blind,
        "manual_semantic": args.manual_semantic,
        "manual_output_audit": args.manual_output_audit,
    }
    rows, report = evaluate(
        json.loads(args.preregistration.read_text()),
        read_jsonl(args.sealed),
        read_jsonl(args.storyboards),
        read_jsonl(args.qwen),
        read_jsonl(args.gemma),
        read_tsv(args.manual_blind),
        read_tsv(args.manual_semantic),
        read_tsv(args.manual_output_audit),
    )
    args.rows_out.parent.mkdir(parents=True, exist_ok=True)
    args.rows_out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    report["artifact_sha256"] = {name: sha256(path) for name, path in inputs.items()}
    report["artifact_sha256"]["normalized_rows"] = sha256(args.rows_out)
    args.summary_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
