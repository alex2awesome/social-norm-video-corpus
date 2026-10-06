#!/usr/bin/env python3
"""Export a frozen instructional audit as a scene-mechanism benchmark.

The exporter performs only deterministic joins and validation. It never
changes corpus media or metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def final_blind_decisions(
    frame_rows: list[dict], resolution_rows: list[dict]
) -> dict[tuple[int, str], bool]:
    def frame_decision(row: dict) -> str:
        decision = row.get("blind_scene_presence", row.get("blind_scene"))
        if decision is None:
            raise ValueError(
                "blind review row must contain blind_scene_presence or blind_scene"
            )
        return decision

    def resolved_decision(row: dict) -> str:
        decision = row.get(
            "resolved_scene_presence", row.get("resolved_blind_scene")
        )
        if decision is None:
            raise ValueError(
                "motion row must contain resolved_scene_presence or "
                "resolved_blind_scene"
            )
        return decision

    resolutions = {
        (row["audit_index"], row["uid"]): resolved_decision(row)
        for row in resolution_rows
    }
    expected = {
        (row["audit_index"], row["uid"])
        for row in frame_rows
        if frame_decision(row) == "needs_motion"
    }
    if set(resolutions) != expected:
        raise ValueError("motion resolutions must exactly cover needs_motion rows")

    result = {}
    for row in frame_rows:
        key = (row["audit_index"], row["uid"])
        decision = frame_decision(row)
        if decision == "needs_motion":
            decision = resolutions[key]
        if decision not in {"yes", "no"}:
            raise ValueError(f"invalid final scene decision for {key}: {decision}")
        result[key] = decision == "yes"
    return result


def export_rows(
    manifest: dict,
    frame_rows: list[dict],
    resolution_rows: list[dict],
    label_rows: list[dict],
    clips_dir: Path,
) -> list[dict]:
    source_rows = manifest["visual_samples"]
    identity = [(row["audit_index"], row["uid"]) for row in source_rows]
    if identity != [
        (row["audit_index"], row["uid"]) for row in frame_rows
    ] or identity != [
        (row["audit_index"], row["uid"]) for row in label_rows
    ]:
        raise ValueError("manual artifacts do not exactly match manifest identity/order")

    blind = final_blind_decisions(frame_rows, resolution_rows)
    exported = []
    for source, review in zip(source_rows, label_rows):
        index = source["audit_index"]
        uid = source["uid"]
        key = (index, uid)
        disposition = review.get("disposition", review.get("adjudication"))
        if disposition is None:
            raise ValueError(
                f"label review must contain disposition or adjudication for {key}"
            )
        proposed_norm = review.get("proposed_norm", source["norm"])
        if proposed_norm != source["norm"]:
            raise ValueError(f"proposed norm drift for {key}")
        relabel = review.get("relabel", review.get("relabel_norm"))
        if (disposition == "accept_relabel") != (relabel is not None):
            raise ValueError(f"invalid relabel contract for {key}")
        if "final_scene" in review:
            expected_scene = "yes" if blind[key] else "no"
            if review["final_scene"] != expected_scene:
                raise ValueError(f"label/blind scene drift for {key}")
            if disposition in {"accept_exact", "accept_relabel"} and not blind[key]:
                raise ValueError(f"cannot accept a visually absent scene for {key}")
        relabel_polarity = review.get("relabel_polarity")
        if relabel_polarity is not None and relabel_polarity not in {
            "violation",
            "correct",
            "explanation",
        }:
            raise ValueError(f"invalid relabel polarity for {key}")

        filename = f"{index:03d}_{uid}.mp4"
        row = {
            "item_id": f"instructional:{uid}:{source['clip_index']}",
            "audit_index": index,
            "pillar": "instructional",
            "uid": uid,
            "title": source.get("title"),
            "category": source.get("category"),
            "polarity": source.get("polarity"),
            "norm": source.get("norm"),
            "explanation": source.get("explanation"),
            "start_quote": source.get("start_quote"),
            "end_quote": source.get("end_quote"),
            "source_clip": source["clip_path"],
            "proxy_clip": str(clips_dir / filename),
            "gold_scene_visible": blind[key],
            "gold_social_scene_visible": (
                blind[key] and disposition != "reject_nonsocial"
            ),
            "gold_label_matched_visible": disposition == "accept_exact",
            "gold_usable": disposition in {"accept_exact", "accept_relabel"},
            "gold_disposition": disposition,
            "gold_relabel": relabel,
            "gold_relabel_polarity": relabel_polarity,
            "is_social_norm": (
                "no" if disposition == "reject_nonsocial" else "yes"
            ),
            "source_recut_candidate": bool(review["source_recut_candidate"]),
            "gold_rubric": "instructional_fresh_blind_then_label_v1",
        }
        exported.append(row)
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--blind-review", type=Path, required=True)
    parser.add_argument("--motion-resolution", type=Path, required=True)
    parser.add_argument("--label-review", type=Path, required=True)
    parser.add_argument("--clips-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--remote-files-out", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    rows = export_rows(
        manifest,
        load_jsonl(args.blind_review),
        load_jsonl(args.motion_resolution),
        load_jsonl(args.label_review),
        args.clips_dir,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    if args.remote_files_out:
        args.remote_files_out.write_text(
            "".join(row["source_clip"] + "\n" for row in rows)
        )
    print(json.dumps({
        "items": len(rows),
        "scene_visible": sum(row["gold_scene_visible"] for row in rows),
        "social_scene_visible": sum(
            row["gold_social_scene_visible"] for row in rows
        ),
        "label_matched_visible": sum(
            row["gold_label_matched_visible"] for row in rows
        ),
        "usable": sum(row["gold_usable"] for row in rows),
        "source_recut_candidates": sum(
            row["source_recut_candidate"] for row in rows
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
