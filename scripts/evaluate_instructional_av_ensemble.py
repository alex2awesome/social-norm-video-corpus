#!/usr/bin/env python3
"""Evaluate transcript + video-VLM shadow rules against complete manual gold.

The inputs may be append-only JSONL with failed attempts followed by retries.
Only the latest successful judgment per item is used.  The evaluator refuses
partial or identity-mismatched runs so headline metrics cannot silently omit
hard examples.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unique_by_item(records: list[dict], *, source: str) -> dict[str, dict]:
    """Keep the latest successful append-only record for each item."""
    successful: dict[str, dict] = {}
    for row in records:
        if row.get("result") is not None and row.get("error") is None:
            successful[row["item_id"]] = row
    if not successful:
        raise ValueError(f"{source} contains no successful predictions")
    return successful


def primary_with_fallbacks(
    primary: list[dict],
    fallbacks: list[list[dict]],
    *,
    source: str,
) -> list[dict]:
    """Fill missing primary predictions without overriding higher-quality runs."""
    merged = unique_by_item(primary, source=source)
    for index, records in enumerate(fallbacks):
        fallback = unique_by_item(records, source=f"{source} fallback {index}")
        for item_id, row in fallback.items():
            merged.setdefault(item_id, row)
    return list(merged.values())


def manual_by_item(records: list[dict]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for row in records:
        item_id = row["item_id"]
        if item_id in result:
            raise ValueError(f"duplicate manual item_id: {item_id}")
        result[item_id] = row
    return result


def manual_gold(row: dict) -> dict[str, bool]:
    visual = row["visual_demo_present"] == "yes"
    social = row["is_social_norm"] == "yes"
    label = row["norm_supported"] == "yes"
    return {
        "keep_after_relabel": visual and social,
        "strict_current_label": visual and social and label,
    }


def yes(row: dict, field: str) -> bool:
    return row["result"].get(field) == "yes"


def not_no(row: dict, field: str) -> bool:
    """Treat transcript uncertainty as abstention, not negative evidence."""
    return row["result"].get(field) != "no"


def feature_row(
    qwen: dict,
    glm: dict,
    transcript: dict,
    *,
    video_rubric: str = "v3",
) -> dict[str, bool]:
    if video_rubric == "v4":
        q_scene = yes(qwen, "usable_demo_after_relabel")
        q_label = yes(qwen, "proposed_norm_supported")
        g_scene = yes(glm, "usable_demo_after_relabel")
        g_label = yes(glm, "proposed_norm_supported")
        q_domain = yes(qwen, "social_norm_domain")
        g_domain = yes(glm, "social_norm_domain")
    else:
        q_scene = yes(qwen, "situated_social_scenario_visible")
        q_label = yes(qwen, "proposed_norm_plausibly_demonstrated")
        g_scene = yes(glm, "situated_social_scenario_visible")
        g_label = yes(glm, "proposed_norm_plausibly_demonstrated")
        q_domain = g_domain = True
    t_social = yes(transcript, "social_norm_topic_supported")
    t_specific = yes(transcript, "specific_hypothesis_supported")
    t_social_not_no = not_no(transcript, "social_norm_topic_supported")
    t_specific_not_no = not_no(transcript, "specific_hypothesis_supported")
    t_example = yes(transcript, "demonstration_or_example_language_present")
    t_situated = yes(transcript, "situated_dialogue_or_action_cues_present")
    t_discussion_only = yes(transcript, "discussion_or_advice_only")
    return {
        "qwen_scene": q_scene,
        "qwen_label": q_label,
        "qwen_domain": q_domain,
        "glm_scene": g_scene,
        "glm_label": g_label,
        "glm_domain": g_domain,
        "transcript_social": t_social,
        "transcript_specific": t_specific,
        "transcript_social_not_no": t_social_not_no,
        "transcript_specific_not_no": t_specific_not_no,
        "transcript_example": t_example,
        "transcript_situated": t_situated,
        "transcript_discussion_only": t_discussion_only,
        "dual_scene": q_scene and g_scene,
        "dual_label": q_label and g_label,
    }


Rule = Callable[[dict[str, bool]], bool]


RULES: dict[str, Rule] = {
    "qwen_scene": lambda f: f["qwen_scene"],
    "glm_scene": lambda f: f["glm_scene"],
    "dual_scene": lambda f: f["dual_scene"],
    "dual_scene_dual_domain": lambda f: (
        f["dual_scene"] and f["qwen_domain"] and f["glm_domain"]
    ),
    "dual_scene_transcript_social": lambda f: (
        f["dual_scene"] and f["transcript_social"]
    ),
    "qwen_scene_transcript_topic_veto": lambda f: (
        f["qwen_scene"] and f["transcript_social_not_no"]
    ),
    "glm_scene_transcript_topic_veto": lambda f: (
        f["glm_scene"] and f["transcript_social_not_no"]
    ),
    "dual_scene_transcript_topic_veto": lambda f: (
        f["dual_scene"] and f["transcript_social_not_no"]
    ),
    "dual_scene_transcript_topic_and_presentation_veto": lambda f: (
        f["dual_scene"]
        and f["transcript_social_not_no"]
        and not f["transcript_discussion_only"]
    ),
    "dual_scene_transcript_topic_specific_veto": lambda f: (
        f["dual_scene"]
        and f["transcript_social_not_no"]
        and f["transcript_specific_not_no"]
    ),
    "dual_scene_transcript_social_specific": lambda f: (
        f["dual_scene"] and f["transcript_social"] and f["transcript_specific"]
    ),
    "dual_scene_transcript_scenario_cue": lambda f: (
        f["dual_scene"]
        and f["transcript_social"]
        and (f["transcript_example"] or f["transcript_situated"])
        and not f["transcript_discussion_only"]
    ),
    "strict_av_label_agreement": lambda f: (
        f["dual_scene"]
        and f["dual_label"]
        and f["transcript_social"]
        and f["transcript_specific"]
    ),
    "strict_av_label_with_transcript_veto": lambda f: (
        f["dual_scene"]
        and f["dual_label"]
        and f["transcript_social_not_no"]
        and f["transcript_specific_not_no"]
    ),
}


def metrics(rows: list[dict], target: str, rule: Rule) -> dict:
    tp = fp = tn = fn = 0
    disagreements = []
    for row in rows:
        gold = row["gold"][target]
        prediction = rule(row["features"])
        if gold and prediction:
            tp += 1
        elif not gold and prediction:
            fp += 1
            disagreements.append(disagreement(row, "fp"))
        elif not gold and not prediction:
            tn += 1
        else:
            fn += 1
            disagreements.append(disagreement(row, "fn"))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "n": len(rows),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and precision + recall
            else None
        ),
        "disagreements": disagreements,
    }


def disagreement(row: dict, outcome: str) -> dict:
    return {
        "item_id": row["item_id"],
        "outcome": outcome,
        "manual_decision": row["manual"]["decision"],
        "manual_evidence": row["manual"]["evidence"],
        "features": row["features"],
        "qwen_evidence": row["qwen"]["result"].get("evidence"),
        "glm_evidence": row["glm"]["result"].get("evidence"),
        "transcript_evidence": row["transcript"]["result"].get("evidence"),
    }


def evaluate(
    manual_records: list[dict],
    qwen_records: list[dict],
    glm_records: list[dict],
    transcript_records: list[dict],
    *,
    video_rubric: str = "v3",
) -> dict:
    manual = manual_by_item(manual_records)
    sources = {
        "qwen": unique_by_item(qwen_records, source="qwen"),
        "glm": unique_by_item(glm_records, source="glm"),
        "transcript": unique_by_item(transcript_records, source="transcript"),
    }
    expected = set(manual)
    identity = {}
    for name, records in sources.items():
        missing = sorted(expected - records.keys())
        extra = sorted(records.keys() - expected)
        identity[name] = {
            "successful": len(records),
            "missing": missing,
            "extra": extra,
        }
        if missing or extra:
            raise ValueError(
                f"{name} identity mismatch: missing={len(missing)} extra={len(extra)}"
            )

    joined = []
    for item_id, manual_row in manual.items():
        qwen = sources["qwen"][item_id]
        glm = sources["glm"][item_id]
        transcript = sources["transcript"][item_id]
        joined.append(
            {
                "item_id": item_id,
                "manual": manual_row,
                "qwen": qwen,
                "glm": glm,
                "transcript": transcript,
                "gold": manual_gold(manual_row),
                "features": feature_row(
                    qwen,
                    glm,
                    transcript,
                    video_rubric=video_rubric,
                ),
            }
        )

    return {
        "complete": True,
        "video_rubric": video_rubric,
        "manual_items": len(manual),
        "identity": identity,
        "gold_counts": {
            target: sum(row["gold"][target] for row in joined)
            for target in ("keep_after_relabel", "strict_current_label")
        },
        "rules": {
            name: {
                target: metrics(joined, target, rule)
                for target in ("keep_after_relabel", "strict_current_label")
            }
            for name, rule in RULES.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument(
        "--glm-fallback",
        type=Path,
        action="append",
        default=[],
        help="Fill failed/missing primary GLM items without overriding successes.",
    )
    parser.add_argument("--transcript", type=Path, required=True)
    parser.add_argument("--video-rubric", choices=("v3", "v4"), default="v3")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(
        load_jsonl(args.manual),
        load_jsonl(args.qwen),
        primary_with_fallbacks(
            load_jsonl(args.glm),
            [load_jsonl(path) for path in args.glm_fallback],
            source="glm",
        ),
        load_jsonl(args.transcript),
        video_rubric=args.video_rubric,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "complete": report["complete"],
                "manual_items": report["manual_items"],
                "gold_counts": report["gold_counts"],
                "out": str(args.out),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
