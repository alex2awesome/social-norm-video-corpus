#!/usr/bin/env python3
"""Evaluate frozen instructional metadata/script cues against manual labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from math import sqrt
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_instructional_scene_title_cues import title_cues
except ModuleNotFoundError:
    from evaluate_instructional_scene_title_cues import title_cues  # type: ignore[no-redef]


FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|i'll|me|my|mine|we|we're|us|our|ours)\b", re.I)
SECOND_PERSON = re.compile(r"\b(?:you|you're|you've|you'll|your|yours)\b", re.I)
SOCIAL_ACTION = re.compile(
    r"\b(?:apolog(?:ize|ise|ized|ised|izing|ising)|ask(?:ed|ing)?|"
    r"attack(?:ed|ing)?|beat(?:en|ing)?|bull(?:y|ied|ying)|cheat(?:ed|ing)?|"
    r"comfort(?:ed|ing)?|confront(?:ed|ing)?|exclude(?:d|s|ing)?|"
    r"fight(?:ing)?|forgiv(?:e|es|ing|en)|grab(?:bed|bing)?|help(?:ed|ing)?|"
    r"hit(?:ting)?|hug(?:ged|ging)?|ignore(?:d|s|ing)?|insult(?:ed|ing)?|"
    r"interrupt(?:ed|ing)?|invite(?:d|s|ing)?|lie|lied|lying|mock(?:ed|ing)?|"
    r"push(?:ed|ing)?|refus(?:e|ed|es|ing)|respect(?:ed|ing)?|"
    r"share(?:d|s|ing)?|shout(?:ed|ing)?|steal|stole|stolen|stealing|"
    r"thank(?:ed|ing)?|threaten(?:ed|ing)?|touch(?:ed|ing)?|"
    r"yell(?:ed|ing)?)\b",
    re.I,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [center - half, center + half]


def cue_values(row: dict[str, Any]) -> dict[str, bool]:
    title_scene = bool(title_cues(str(row.get("title") or ""))["scene_title_candidate"])
    query_scene = bool(
        title_cues(str(row.get("found_by_query") or ""))["scene_title_candidate"]
    )
    quote_text = " ".join(
        str(row.get(field) or "")
        for field in ("start_quote", "end_quote", "explanation")
    )
    non_explanation = str(row.get("polarity") or "").strip().lower() != "explanation"
    dialogue_exchange = bool(FIRST_PERSON.search(quote_text) and SECOND_PERSON.search(quote_text))
    social_action = bool(SOCIAL_ACTION.search(quote_text))
    title_or_query = title_scene or query_scene
    return {
        "title_scene": title_scene,
        "query_scene": query_scene,
        "title_or_query_scene": title_or_query,
        "non_explanation": non_explanation,
        "dialogue_exchange": dialogue_exchange,
        "social_action_lexicon": social_action,
        "scene_and_non_explanation": title_or_query and non_explanation,
    }


def metric(rows: list[dict[str, Any]], cue: str, target: str) -> dict[str, Any]:
    selected = [row for row in rows if row["cues"][cue]]
    positives = [row for row in rows if row[target]]
    tp = sum(bool(row[target]) for row in selected)
    fp = len(selected) - tp
    fn = len(positives) - tp
    precision = tp / len(selected) if selected else None
    base = len(positives) / len(rows) if rows else None
    return {
        "n": len(rows),
        "manual_positives": len(positives),
        "selected": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "precision_wilson_95": wilson(tp, len(selected)),
        "recall": tp / len(positives) if positives else None,
        "base_rate": base,
        "precision_delta_over_base": precision - base if precision is not None and base is not None else None,
    }


def evaluate(rows: list[dict[str, Any]], prereg: dict[str, Any]) -> dict[str, Any]:
    cues = list(prereg["cues"])
    targets = {"visual_demo": "manual_visual_demo", "exact_demo": "manual_exact_original"}
    metrics = {
        cue: {name: metric(rows, cue, field) for name, field in targets.items()}
        for cue in cues
    }
    gate = prereg["evaluation"]
    decisions = {}
    for cue in cues:
        visual = metrics[cue]["visual_demo"]
        passed = (
            visual["selected"] >= gate["minimum_selected"]
            and visual["precision"] is not None
            and visual["precision"] >= gate["ranking_gate_visual_precision"]
            and visual["precision_delta_over_base"] >= gate["ranking_gate_visual_precision_delta_over_base"]
            and visual["recall"] >= gate["ranking_gate_visual_recall"]
        )
        decisions[cue] = {
            "ranking_gate_passed": passed,
            "allowed_use": "manual_review_priority_only" if passed else "untrusted_diagnostic_only",
            "automatic_acceptance": False,
        }
    return {
        "kind": "instructional_metadata_script_cues_v1_evaluation",
        "policy": prereg["policy"],
        "items": len(rows),
        "unique_uids": len({row["uid"] for row in rows}),
        "metrics": metrics,
        "decisions": decisions,
        "all_outputs_manually_scored": len(rows),
        "automatic_acceptance": False,
        "corpus_mutated": False,
    }


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(raw["candidate_id"]),
        "item_id": str(raw["item_id"]),
        "uid": str(raw["uid"]),
        "title": str(raw.get("title") or ""),
        "found_by_query": str(raw.get("found_by_query") or ""),
        "polarity": str(raw.get("polarity") or ""),
        "manual_visual_demo": bool(raw["manual_visual_demo"]),
        "manual_exact_original": bool(raw["manual_exact_original"]),
        "visual_form": str(raw.get("visual_form") or ""),
        "manual_note": str(raw.get("blind_visual_note") or ""),
        "cues": cue_values(raw),
    }


def run(input_path: Path, prereg_path: Path, output: Path, summary_path: Path) -> dict[str, Any]:
    prereg = json.loads(prereg_path.read_text())
    expected = prereg["evaluation_cohort"]["records_sha256"]
    actual = sha256(input_path)
    if actual != expected:
        raise ValueError(f"evaluation input hash mismatch: {actual} != {expected}")
    rows = [normalize(row) for row in read_jsonl(input_path)]
    expected_n = int(prereg["evaluation_cohort"]["items"])
    if len(rows) != expected_n or len({row["uid"] for row in rows}) != expected_n:
        raise ValueError("evaluation cohort size or source independence changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    summary = evaluate(rows, prereg)
    summary["artifact_sha256"] = {
        "preregistration": sha256(prereg_path),
        "input_manual_records": actual,
        "normalized_records": sha256(output),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input, args.preregistration, args.output, args.summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
