#!/usr/bin/env python3
"""Adapt generic video baselines to commentary window-ranking features.

This is a scale-free candidate-ranking adapter, not a keep classifier. It maps
bounded-window low-level, keypoint, CLIP, and X-CLIP outputs to the four feature
names consumed by ``build_commentary_hierarchical_window_manifest_v2.py``.
Missing model sections produce an explicit error and therefore rank below
successful windows; they never become implicit negative labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.build_commentary_hierarchical_window_manifest_v2 import (
        read_jsonl,
        write_jsonl,
    )
else:
    from build_commentary_hierarchical_window_manifest_v2 import read_jsonl, write_jsonl


CLIP_SOCIAL_PROMPT = "an actual social interaction or confrontation caught on camera"
XCLIP_SOCIAL_PROMPTS = {
    "a situated social interaction between two or more people",
    "a visible interpersonal conflict, norm violation, or confrontation",
    "people acting out a social situation or role play",
}


def prompt_score(
    section: dict[str, Any],
    accepted_prompts: set[str],
    probability_field: str,
) -> float:
    prompts = section.get("prompts")
    probabilities = section.get(probability_field)
    if (
        not isinstance(prompts, list)
        or not isinstance(probabilities, list)
        or len(prompts) != len(probabilities)
    ):
        raise ValueError("invalid prompt/probability arrays")
    values = [
        float(probabilities[index])
        for index, prompt in enumerate(prompts)
        if prompt in accepted_prompts
    ]
    if not values:
        raise ValueError("required social-scene prompt is absent")
    return max(values)


def adapt_one(row: dict[str, Any]) -> dict[str, Any]:
    window_id = str(row.get("item_id") or "")
    result: dict[str, Any] = {"window_id": window_id}
    if not window_id:
        return {"window_id": "", "error": "ValueError: missing item_id"}
    if row.get("error"):
        return {"window_id": window_id, "error": str(row["error"])}
    try:
        low = row.get("low_level")
        keypoints = row.get("keypoints")
        if not isinstance(low, dict) or not isinstance(keypoints, dict):
            raise ValueError("low_level and keypoints are required")
        motion = float(low["motion_mean"])
        scene_change = max(
            float(low["histogram_delta_mean"]),
            float(low["hard_cut_fraction"]),
        )
        person_interaction = max(
            float(keypoints.get("multiple_people_fraction") or 0),
            float(keypoints.get("close_pair_fraction") or 0),
            float(keypoints.get("wrist_near_other_person_fraction") or 0),
        )
        similarities = []
        if isinstance(row.get("xclip_scores"), dict):
            similarities.append(prompt_score(
                row["xclip_scores"], XCLIP_SOCIAL_PROMPTS, "probabilities"
            ))
        if isinstance(row.get("clip_scores"), dict):
            similarities.append(prompt_score(
                row["clip_scores"], {CLIP_SOCIAL_PROMPT}, "mean_probabilities"
            ))
        if not similarities:
            raise ValueError("CLIP or X-CLIP social-scene scores are required")
        result.update({
            "motion": motion,
            "scene_change": scene_change,
            "person_interaction": person_interaction,
            "social_scene_similarity": max(similarities),
            "error": None,
            "policy": "candidate_ranking_only_no_keep_reject_or_corpus_mutation",
        })
    except (KeyError, TypeError, ValueError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def adapt(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = [adapt_one(row) for row in rows]
    ids = [row["window_id"] for row in output]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("baseline rows have missing or duplicate item_id")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_jsonl(args.out, adapt(read_jsonl(args.baseline_scores)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
