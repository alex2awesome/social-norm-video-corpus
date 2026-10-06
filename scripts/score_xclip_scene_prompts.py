#!/usr/bin/env python3
"""Score frozen video proxies with a temporal X-CLIP prompt bank.

This is a shadow feature extractor. It writes append-only JSONL and never
changes source clips or corpus metadata.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import XCLIPModel, XCLIPProcessor


PROMPTS = [
    "a situated scene where characters interact with each other",
    "a dialogue scene between characters inside a shared situation",
    "a staged role-play demonstrating social behavior",
    "a person performing a socially meaningful action in a shared public context",
    "a visible norm violation or correction happening on screen",
    "a commentator speaking directly to the camera",
    "a lecture, interview, or audience-directed talking head",
    "news footage or unrelated illustrative b-roll",
    "a slide, chart, diagram, or text graphic",
    "an empty setting or context with no social event",
]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def resolve_clip(row: dict, manifest: Path) -> Path:
    raw = Path(row.get("proxy_clip") or row["source_clip"])
    local = manifest.parent / "clips" / raw.name
    return local if local.exists() else raw


def read_uniform_frames(path: Path, count: int) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    frame_count = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    indexes = np.linspace(0, frame_count - 1, count).round().astype(int)
    frames: list[np.ndarray] = []
    wanted = set(indexes.tolist())
    index = 0
    while capture.isOpened() and wanted:
        ok, frame = capture.read()
        if not ok:
            break
        if index in wanted:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            wanted.remove(index)
        index += 1
    capture.release()
    if not frames:
        raise ValueError(f"decoded no frames from {path}")
    while len(frames) < count:
        frames.append(frames[-1].copy())
    return frames[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="microsoft/xclip-base-patch32")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = load_jsonl(args.manifest)
    if args.limit is not None:
        rows = rows[: args.limit]
    completed = {
        row["item_id"]
        for row in load_jsonl(args.out)
    } if args.out.exists() else set()
    pending = [row for row in rows if row["item_id"] not in completed]

    processor = XCLIPProcessor.from_pretrained(args.model)
    model = XCLIPModel.from_pretrained(args.model).to(args.device).eval()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a") as handle:
        for index, row in enumerate(pending, 1):
            error = None
            logits = probabilities = None
            try:
                frames = read_uniform_frames(
                    resolve_clip(row, args.manifest), args.frames
                )
                inputs = processor(
                    text=PROMPTS,
                    videos=frames,
                    return_tensors="pt",
                    padding=True,
                )
                inputs = {
                    key: value.to(args.device)
                    for key, value in inputs.items()
                }
                with torch.inference_mode():
                    output = model(**inputs)
                values = output.logits_per_video[0].float().cpu()
                logits = values.tolist()
                probabilities = values.softmax(dim=-1).tolist()
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            result = {
                "item_id": row["item_id"],
                "uid": row["uid"],
                "pillar": row["pillar"],
                "gold_scene_visible": row["gold_scene_visible"],
                "gold_social_scene_visible": row["gold_social_scene_visible"],
                "gold_label_matched_visible": row["gold_label_matched_visible"],
                "model": args.model,
                "frames": args.frames,
                "xclip_scores": {
                    "prompts": PROMPTS,
                    "logits": logits,
                    "probabilities": probabilities,
                },
                "error": error,
            }
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"{index}/{len(pending)} {row['item_id']} "
                f"{'ok' if error is None else error}",
                flush=True,
            )


if __name__ == "__main__":
    main()
