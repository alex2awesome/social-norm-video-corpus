#!/usr/bin/env python3
"""Build a frozen, conditioned VLM manifest from a completed human calibration.

The output only joins existing audit artifacts.  It does not mutate corpus
metadata, labels, clips, or routing state.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def indexed(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(row["audit_index"]): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate audit_index")
    return result


def media_bounds(selection: dict[str, Any]) -> tuple[float | None, float | None]:
    if selection["pillar"] != "commentary":
        return None, None
    start = selection.get("start_sec")
    end = selection.get("end_sec")
    if start is None or end is None:
        return None, None
    return max(0.0, float(start) - 12.0), float(end) + 12.0


def transcript_origin(
    selection: dict[str, Any], semantic: dict[str, Any], media_start: float | None
) -> float:
    if media_start is not None:
        return media_start
    if selection["pillar"] == "instructional":
        demo = semantic.get("semantic", {}).get("demo", {})
        return float(demo.get("start") or 0.0)
    if selection["pillar"] == "witnessed":
        return float(selection.get("start_sec") or 0.0)
    return 0.0


def aligned_transcript(
    semantic: dict[str, Any], origin: float
) -> list[dict[str, Any]]:
    aligned = []
    for segment in semantic.get("transcript_segments") or []:
        try:
            start = max(0.0, float(segment["start"]) - origin)
            end = max(start, float(segment["end"]) - origin)
        except (KeyError, TypeError, ValueError):
            continue
        aligned.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": str(segment.get("text") or "").strip(),
            }
        )
    return aligned


def detector_explanation(selection: dict[str, Any], semantic: dict[str, Any]) -> str:
    payload = semantic.get("semantic", {})
    if selection["pillar"] == "instructional":
        return str(payload.get("demo", {}).get("explanation") or "")
    if selection["pillar"] == "witnessed":
        values = [selection.get("reaction_tag"), selection.get("reaction_text")]
        return ": ".join(str(value) for value in values if value)
    statement = payload.get("statement", {})
    return str(statement.get("quote") or selection.get("quote") or "")


def build_rows(
    root: Path,
    selections: list[dict[str, Any]],
    semantics: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    semantic_by_index = indexed(semantics)
    review_by_index = indexed(reviews)
    if set(semantic_by_index) != set(review_by_index):
        raise ValueError("semantic/review audit indices differ")
    result = []
    for ordinal, selection in enumerate(sorted(selections, key=lambda row: row["audit_index"])):
        audit_index = int(selection["audit_index"])
        semantic = semantic_by_index[audit_index]
        review = review_by_index[audit_index]
        if semantic["item_id"] != selection["item_id"]:
            raise ValueError(f"item mismatch at audit index {audit_index}")
        if review["pillar"] != selection["pillar"]:
            raise ValueError(f"pillar mismatch at audit index {audit_index}")
        start, end = media_bounds(selection)
        origin = transcript_origin(selection, semantic, start)
        social_scene = review["social_norm_scene"] == "yes"
        exact_label = review["label_alignment"] == "yes"
        strict_decision = str(review["strict_decision"])
        result.append(
            {
                "ordinal": ordinal,
                "audit_index": audit_index,
                "item_id": selection["item_id"],
                "uid": selection["uid"],
                "pillar": selection["pillar"],
                "source_clip": str((root / selection["media_path"]).resolve()),
                "media_start_sec": start,
                "media_end_sec": end,
                "duration_hint": (
                    round(end - start, 3)
                    if start is not None and end is not None
                    else None
                ),
                "norm": selection.get("norm") or "unspecified",
                "explanation": detector_explanation(selection, semantic),
                "aligned_transcript": aligned_transcript(semantic, origin),
                "gold_scene_visible": social_scene,
                "gold_social_scene_visible": social_scene,
                "gold_label_matched_visible": exact_label,
                "gold_usable": strict_decision == "accept",
                "gold_repairable": strict_decision == "accept_with_repairs",
                "gold_strict_decision": strict_decision,
                "gold_salvage_route": review["salvage_route"],
                "gold_label_alignment": review["label_alignment"],
                "gold_witnessed_strict": (
                    selection["pillar"] == "witnessed"
                    and strict_decision == "accept"
                ),
            }
        )
    if len({row["item_id"] for row in result}) != len(result):
        raise ValueError("duplicate item_id")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--semantics", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = build_rows(
        args.root,
        load_jsonl(args.selection),
        load_jsonl(args.semantics),
        load_jsonl(args.reviews),
    )
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if args.out.exists() and args.out.read_text() != content:
        raise FileExistsError(f"refusing to overwrite different frozen manifest: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(content)
    print(json.dumps({"items": len(rows), "out": str(args.out)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
