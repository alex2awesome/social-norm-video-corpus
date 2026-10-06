#!/usr/bin/env python3
"""Build guarded recurrent multilingual OCR-line masks for commentary clips."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

if __package__:
    from scripts.build_commentary_ocr_mask_plan import sample_sequential_frames, sha256
    from scripts.materialize_commentary_caption_crop_experiment import read_jsonl, read_tsv
    from scripts.materialize_commentary_motion_crop_experiment import derive_parents
else:
    from build_commentary_ocr_mask_plan import sample_sequential_frames, sha256
    from materialize_commentary_caption_crop_experiment import read_jsonl, read_tsv
    from materialize_commentary_motion_crop_experiment import derive_parents


def detect_text_line_boxes(
    frame: np.ndarray, languages: str = "eng+hin+guj", minimum_confidence: float = 40,
) -> list[tuple[int, int, int, int]]:
    data = pytesseract.image_to_data(
        frame, lang=languages, config="--psm 11", output_type=Output.DICT
    )
    groups: dict[tuple[int, int, int], list[tuple[int, int, int, int]]] = {}
    for index, text in enumerate(data["text"]):
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if confidence < minimum_confidence or not str(text).strip():
            continue
        width = int(data["width"][index])
        height = int(data["height"][index])
        if width < 3 or height < 3:
            continue
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        groups.setdefault(key, []).append((
            int(data["left"][index]), int(data["top"][index]), width, height
        ))
    lines = []
    for words in groups.values():
        x0 = min(word[0] for word in words)
        y0 = min(word[1] for word in words)
        x1 = max(word[0] + word[2] for word in words)
        y1 = max(word[1] + word[3] for word in words)
        lines.append((x0, y0, x1 - x0, y1 - y0))
    return lines


def compatible_line(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int],
    frame_width: int, frame_height: int,
) -> bool:
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    vertical_overlap = max(0, min(ay + ah, by + bh) - max(ay, by))
    horizontal_overlap = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    vertical_ratio = vertical_overlap / max(1, min(ah, bh))
    horizontal_ratio = horizontal_overlap / max(1, min(aw, bw))
    center_y_close = abs((ay + ah / 2) - (by + bh / 2)) <= frame_height * 0.04
    edge_alignment = min(abs(ax - bx), abs((ax + aw) - (bx + bw))) <= frame_width * 0.05
    return center_y_close and vertical_ratio >= 0.35 and (
        horizontal_ratio >= 0.20 or edge_alignment
    )


def recurrent_line_boxes(
    lines_by_frame: list[list[tuple[int, int, int, int]]],
    frame_width: int,
    frame_height: int,
    minimum_frames: int = 2,
    maximum_box_area: float = 0.15,
    maximum_total_area: float = 0.25,
) -> list[tuple[float, float, float, float]]:
    clusters: list[dict[str, Any]] = []
    for frame_index, lines in enumerate(lines_by_frame):
        used: set[int] = set()
        for line in lines:
            candidates = [
                (index, cluster) for index, cluster in enumerate(clusters)
                if index not in used
                and frame_index not in cluster["frames"]
                and compatible_line(
                    cluster["boxes"][-1], line, frame_width, frame_height
                )
            ]
            if candidates:
                index, cluster = min(
                    candidates,
                    key=lambda value: abs(
                        value[1]["boxes"][-1][1] - line[1]
                    ),
                )
                cluster["boxes"].append(line)
                cluster["frames"].add(frame_index)
                used.add(index)
            else:
                clusters.append({"boxes": [line], "frames": {frame_index}})
    normalized: list[tuple[float, float, float, float]] = []
    for cluster in clusters:
        if len(cluster["frames"]) < minimum_frames:
            continue
        boxes = cluster["boxes"]
        x0 = int(np.percentile([box[0] for box in boxes], 10))
        y0 = int(np.percentile([box[1] for box in boxes], 10))
        x1 = int(np.percentile([box[0] + box[2] for box in boxes], 90))
        y1 = int(np.percentile([box[1] + box[3] for box in boxes], 90))
        pad_x = max(2, int(round(frame_width * 0.008)))
        pad_y = max(2, int(round(frame_height * 0.010)))
        x0 = max(0, x0 - pad_x)
        y0 = max(0, y0 - pad_y)
        x1 = min(frame_width, x1 + pad_x)
        y1 = min(frame_height, y1 + pad_y)
        area = (x1 - x0) * (y1 - y0) / (frame_width * frame_height)
        if area <= 0 or area > maximum_box_area:
            continue
        normalized.append((
            x0 / frame_width, y0 / frame_height,
            (x1 - x0) / frame_width, (y1 - y0) / frame_height,
        ))
    if not normalized:
        raise ValueError("no recurrent safe OCR lines")
    union = np.zeros((1000, 1000), dtype=np.uint8)
    for x, y, width, height in normalized:
        x0, y0 = int(x * 1000), int(y * 1000)
        x1, y1 = int(np.ceil((x + width) * 1000)), int(np.ceil((y + height) * 1000))
        union[y0:y1, x0:x1] = 1
    total_area = float(union.mean())
    if total_area > maximum_total_area:
        raise ValueError(f"unsafe total OCR mask area {total_area:.6f}")
    return sorted(normalized, key=lambda box: (box[1], box[0]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sealed", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--post-reveal", type=Path, required=True)
    parser.add_argument("--proxy-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--languages", default="eng+hin+guj")
    args = parser.parse_args()
    if args.out.exists() or args.summary.exists():
        raise SystemExit("refusing to overwrite OCR line-mask plan artifacts")
    sealed = read_jsonl(args.sealed)
    parents = derive_parents(sealed, read_tsv(args.blind), read_tsv(args.post_reveal))
    source_by_id = {row["candidate_id"]: row for row in sealed}
    plans: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for parent in parents:
        source = source_by_id[parent["parent_candidate_id"]]
        proxy = args.proxy_root / f"{int(source['audit_index']):04d}.mp4"
        frames = sample_sequential_frames(proxy, 8)
        height, width = frames[0].shape[:2]
        detections = [
            detect_text_line_boxes(frame, languages=args.languages) for frame in frames
        ]
        try:
            boxes = recurrent_line_boxes(detections, width, height)
        except ValueError as exc:
            failures.append({
                "parent_candidate_id": parent["parent_candidate_id"],
                "reason": str(exc),
            })
            continue
        plans.append({
            "audit_index": len(plans),
            "candidate_id": f"{parent['parent_candidate_id']}--ocr_line_mask_v2",
            "parent_candidate_id": parent["parent_candidate_id"],
            "parent_audit_index": parent["parent_audit_index"],
            "variant": "ocr_line_mask_v2",
            "mask_boxes_normalized": boxes,
            "ocr_languages": args.languages,
            "ocr_frames": len(frames),
            "ocr_line_detections": sum(map(len, detections)),
            "proxy_sha256": sha256(proxy),
            "source_path": parent["source_path"],
            "source_start_sec": parent["source_start_sec"],
            "source_end_sec": parent["source_end_sec"],
            "uid": parent["uid"],
            "title": parent["title"],
            "policy": "shadow_ocr_line_mask_plan_only",
        })
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in plans))
    report = {
        "kind": "commentary_ocr_line_mask_plan_v2",
        "cohort_items": len(parents),
        "renderable_items": len(plans),
        "abstentions": failures,
        "languages": args.languages,
        "ocr_frames_per_item": 8,
        "maximum_box_area": 0.15,
        "maximum_total_area": 0.25,
        "total_mask_boxes": sum(len(row["mask_boxes_normalized"]) for row in plans),
        "plan_sha256": sha256(args.out),
        "automatic_acceptance": False,
        "corpus_mutated": False,
    }
    args.summary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
