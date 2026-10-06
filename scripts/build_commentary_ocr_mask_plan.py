#!/usr/bin/env python3
"""Build deterministic multilingual OCR masks for caption-leaking proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

if __package__:
    from scripts.materialize_commentary_caption_crop_experiment import (
        read_jsonl,
        read_tsv,
    )
    from scripts.materialize_commentary_motion_crop_experiment import derive_parents
else:
    from materialize_commentary_caption_crop_experiment import read_jsonl, read_tsv
    from materialize_commentary_motion_crop_experiment import derive_parents


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_sequential_frames(path: Path, count: int = 8) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) < count:
        raise ValueError(f"{path}: only {len(frames)} decodable frames")
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [frames[int(index)] for index in indices]


def detect_text_boxes(
    frame: np.ndarray, languages: str = "eng+hin+guj", minimum_confidence: float = 20,
) -> list[tuple[int, int, int, int]]:
    data = pytesseract.image_to_data(
        frame, lang=languages, config="--psm 11", output_type=Output.DICT
    )
    boxes: list[tuple[int, int, int, int]] = []
    for index, text in enumerate(data["text"]):
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        width = int(data["width"][index])
        height = int(data["height"][index])
        if confidence < minimum_confidence or not str(text).strip():
            continue
        if width < 3 or height < 3:
            continue
        boxes.append((
            int(data["left"][index]), int(data["top"][index]), width, height
        ))
    return boxes


def persistent_text_boxes(
    boxes_by_frame: list[list[tuple[int, int, int, int]]],
    frame_width: int,
    frame_height: int,
    minimum_frame_fraction: float = 0.20,
) -> list[tuple[float, float, float, float]]:
    """Merge OCR words recurring in the same spatial regions into mask boxes."""
    if not boxes_by_frame:
        raise ValueError("no OCR frames")
    votes = np.zeros((frame_height, frame_width), dtype=np.uint16)
    for boxes in boxes_by_frame:
        frame_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        for x, y, width, height in boxes:
            x0 = max(0, x - 2)
            y0 = max(0, y - 2)
            x1 = min(frame_width, x + width + 2)
            y1 = min(frame_height, y + height + 2)
            if x1 > x0 and y1 > y0:
                frame_mask[y0:y1, x0:x1] = 1
        votes += frame_mask
    required = max(2, math.ceil(len(boxes_by_frame) * minimum_frame_fraction))
    persistent = (votes >= required).astype(np.uint8)
    close_width = max(3, int(round(frame_width * 0.035)))
    close_height = max(3, int(round(frame_height * 0.012)))
    persistent = cv2.morphologyEx(
        persistent, cv2.MORPH_CLOSE,
        np.ones((close_height, close_width), dtype=np.uint8),
    )
    persistent = cv2.dilate(
        persistent,
        np.ones((max(3, int(frame_height * 0.015)), max(3, int(frame_width * 0.01))), dtype=np.uint8),
    )
    count, _, stats, _ = cv2.connectedComponentsWithStats(persistent, 8)
    boxes: list[tuple[float, float, float, float]] = []
    for label in range(1, count):
        x, y, width, height, area = [int(value) for value in stats[label]]
        if area < frame_width * frame_height * 0.00015:
            continue
        if width < frame_width * 0.015 or height < frame_height * 0.012:
            continue
        pad_x = max(2, int(round(frame_width * 0.008)))
        pad_y = max(2, int(round(frame_height * 0.010)))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(frame_width, x + width + pad_x)
        y1 = min(frame_height, y + height + pad_y)
        boxes.append((
            x0 / frame_width, y0 / frame_height,
            (x1 - x0) / frame_width, (y1 - y0) / frame_height,
        ))
    return sorted(boxes, key=lambda box: (box[1], box[0]))


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
        raise SystemExit("refusing to overwrite OCR mask plan artifacts")
    sealed = read_jsonl(args.sealed)
    parents = derive_parents(sealed, read_tsv(args.blind), read_tsv(args.post_reveal))
    source_by_id = {row["candidate_id"]: row for row in sealed}
    plans: list[dict[str, Any]] = []
    for parent in parents:
        source = source_by_id[parent["parent_candidate_id"]]
        proxy = args.proxy_root / f"{int(source['audit_index']):04d}.mp4"
        frames = sample_sequential_frames(proxy, 8)
        height, width = frames[0].shape[:2]
        detections = [
            detect_text_boxes(frame, languages=args.languages) for frame in frames
        ]
        boxes = persistent_text_boxes(detections, width, height)
        if not boxes:
            raise RuntimeError(f"{parent['parent_candidate_id']}: no persistent OCR boxes")
        plans.append({
            "audit_index": len(plans),
            "candidate_id": f"{parent['parent_candidate_id']}--ocr_mask_v1",
            "parent_candidate_id": parent["parent_candidate_id"],
            "parent_audit_index": parent["parent_audit_index"],
            "variant": "ocr_mask_v1",
            "mask_boxes_normalized": boxes,
            "ocr_languages": args.languages,
            "ocr_frames": len(frames),
            "ocr_word_detections": sum(map(len, detections)),
            "proxy_sha256": sha256(proxy),
            "source_path": parent["source_path"],
            "source_start_sec": parent["source_start_sec"],
            "source_end_sec": parent["source_end_sec"],
            "uid": parent["uid"],
            "title": parent["title"],
            "policy": "shadow_ocr_mask_plan_only",
        })
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in plans))
    report = {
        "kind": "commentary_ocr_mask_plan_v1",
        "items": len(plans),
        "languages": args.languages,
        "ocr_frames_per_item": 8,
        "minimum_frame_fraction": 0.20,
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
