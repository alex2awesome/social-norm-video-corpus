#!/usr/bin/env python3
"""Score scene-audit videos with cheap, complementary visual mechanisms.

This produces append-only shadow features.  It does not make a keep/reject
decision and never changes source clips or metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import cv2
import numpy as np


CLIP_PROMPTS = [
    "a staged role play showing a social interaction between people",
    "an actual social interaction or confrontation caught on camera",
    "a person talking directly to the camera",
    "a news anchor, interview, lecture, or podcast",
    "a slide, title card, diagram, or text graphic",
    "generic b-roll that does not show the described action",
]

XCLIP_PROMPTS = [
    "a situated social interaction between two or more people",
    "a visible interpersonal conflict, norm violation, or confrontation",
    "people acting out a social situation or role play",
    "one person speaking directly to the camera",
    "a news report, interview, lecture, or podcast",
    "text, slides, animation, or generic illustrative b-roll",
    "a reaction or aftermath where the triggering behavior is off screen",
]


def xclip_media_kwargs(processor: object, video: list[np.ndarray]) -> dict:
    """Bridge X-CLIP processor versions without guessing from attribute names.

    The published checkpoint may declare an ``image_processor`` attribute even
    when its public call signature requires ``videos``.  Inspecting the call
    contract is therefore more reliable than the internal attribute label.
    """
    call = getattr(processor, "__call__", None)
    if call is not None:
        try:
            if "videos" in inspect.signature(call).parameters:
                return {"videos": video}
        except (TypeError, ValueError):
            pass
    attributes = getattr(processor, "attributes", ())
    if "video_processor" in attributes:
        return {"videos": video}
    return {"images": video}


def xclip_processor_inputs(
    processor: object,
    video: list[np.ndarray],
    **kwargs: object,
) -> object:
    """Process X-CLIP frames across incompatible Transformers contracts.

    Transformers 4.57 exposes ``videos`` in the public signature while the
    checkpoint's ``attributes`` still dispatch only ``images``.  Try the
    advertised contract first, but require actual pixel values and fall back
    to the alternate media keyword when dispatch silently drops the video.
    """
    media = xclip_media_kwargs(processor, video)
    inputs = processor(**kwargs, **media)
    if inputs.get("pixel_values") is not None:
        return inputs
    alternate = (
        {"images": video} if "videos" in media else {"videos": video}
    )
    inputs = processor(**kwargs, **alternate)
    if inputs.get("pixel_values") is None:
        raise ValueError("X-CLIP processor produced no pixel_values")
    return inputs


def uniformly_subsample_frames(
    frames: list[np.ndarray],
    count: int,
) -> list[np.ndarray]:
    if count < 1:
        raise ValueError("frame count must be positive")
    if len(frames) <= count:
        return frames
    indices = np.linspace(0, len(frames) - 1, count, dtype=int)
    return [frames[int(index)] for index in indices]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def shard_for_item(item_id: str, num_shards: int) -> int:
    """Return a stable shard independent of manifest order or Python hash seed."""
    digest = hashlib.sha256(item_id.encode()).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def completed_item_ids(
    paths: list[Path],
    required_sections: tuple[str, ...] = ("low_level",),
) -> set[str]:
    completed = set()
    for path in paths:
        if not path.is_file():
            continue
        for row in load_jsonl(path):
            if (
                row.get("error") is None
                and row.get("item_id")
                and all(row.get(section) for section in required_sections)
            ):
                completed.add(str(row["item_id"]))
    return completed


def resolve_clip(row: dict, manifest: Path) -> Path:
    value = row.get("proxy_clip") or row.get("source_clip") or row["source_path"]
    local = manifest.parent / "clips" / Path(value).name
    return local if local.exists() else Path(value)


def sample_frames(
    path: Path,
    count: int,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> tuple[list[np.ndarray], dict]:
    capture = cv2.VideoCapture(str(path))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total <= 0:
        capture.release()
        return [], {"frame_count": total, "fps": fps, "width": width, "height": height}
    first = 0
    last = max(0, total - 1)
    if fps > 0 and start_sec is not None:
        first = min(last, max(0, int(round(float(start_sec) * fps))))
    if fps > 0 and end_sec is not None:
        last = min(last, max(first, int(round(float(end_sec) * fps)) - 1))
    available = last - first + 1
    indices = np.linspace(first, last, min(count, available), dtype=int)
    frames = []
    sampled_timestamps = []
    bounded = start_sec is not None or end_sec is not None
    if bounded:
        # One seek followed by sequential decode avoids repeatedly entering an
        # H.264 stream between reference frames. A failed target remains
        # missing; callers may show an explicit blank audit tile but must never
        # duplicate a neighboring frame.
        targets = list(map(int, indices))
        target_index = 0
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        for frame_index in range(first, last + 1):
            ok, frame = capture.read()
            if not ok:
                continue
            while (
                target_index < len(targets)
                and frame_index >= targets[target_index]
            ):
                frames.append(frame.copy())
                sampled_timestamps.append(
                    frame_index / fps if fps > 0 else None
                )
                target_index += 1
            if target_index >= len(targets):
                break
    else:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                frames.append(frame)
                sampled_timestamps.append(index / fps if fps > 0 else None)
    capture.release()
    return frames, {
        "frame_count": total,
        "fps": fps,
        "width": width,
        "height": height,
        "duration_sec": total / fps if fps > 0 else None,
        "sampled_frames": len(frames),
        "sampled_timestamps": sampled_timestamps,
        "sample_start_sec": first / fps if fps > 0 else None,
        "sample_end_sec": (last + 1) / fps if fps > 0 else None,
    }


def low_level_features(frames: list[np.ndarray]) -> dict:
    if not frames:
        return {}
    grays = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (320, 180))
        for frame in frames
    ]
    motion = [
        float(np.mean(cv2.absdiff(previous, current)) / 255.0)
        for previous, current in zip(grays, grays[1:])
    ]
    histogram_deltas = []
    for previous, current in zip(frames, frames[1:]):
        left = cv2.calcHist([previous], [0, 1], None, [32, 32], [0, 256, 0, 256])
        right = cv2.calcHist([current], [0, 1], None, [32, 32], [0, 256, 0, 256])
        cv2.normalize(left, left)
        cv2.normalize(right, right)
        histogram_deltas.append(
            float(cv2.compareHist(left, right, cv2.HISTCMP_BHATTACHARYYA))
        )

    face_counts = None
    if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(cascade_path))
        face_counts = []
        for gray in grays:
            faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            face_counts.append(len(faces))
    result = {
        "motion_mean": float(np.mean(motion)) if motion else 0.0,
        "motion_max": float(np.max(motion)) if motion else 0.0,
        "histogram_delta_mean": (
            float(np.mean(histogram_deltas)) if histogram_deltas else 0.0
        ),
        "hard_cut_fraction": (
            float(np.mean(np.asarray(histogram_deltas) > 0.55))
            if histogram_deltas
            else 0.0
        ),
    }
    if face_counts is not None:
        result.update(
            {
                "face_count_mean": float(np.mean(face_counts)),
                "face_present_fraction": float(np.mean(np.asarray(face_counts) > 0)),
                "multiple_faces_fraction": float(
                    np.mean(np.asarray(face_counts) > 1)
                ),
            }
        )
    return result


class KeypointScorer:
    def __init__(self, device: str):
        import torch
        from torchvision.models.detection import (
            KeypointRCNN_ResNet50_FPN_Weights,
            keypointrcnn_resnet50_fpn,
        )

        self.torch = torch
        self.device = device
        self.weights = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = keypointrcnn_resnet50_fpn(weights=self.weights)
        self.model.eval().to(device)
        self.transform = self.weights.transforms()

    def __call__(self, frames: list[np.ndarray]) -> dict:
        images = [
            self.transform(
                self.torch.from_numpy(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                ).permute(2, 0, 1)
            )
            for frame in frames
        ]
        counts, posed, areas = [], [], []
        detections: list[dict[str, Any]] = []
        with self.torch.inference_mode():
            for start in range(0, len(images), 4):
                outputs = self.model(
                    [image.to(self.device) for image in images[start : start + 4]]
                )
                for output in outputs:
                    keep = (output["scores"] >= 0.70) & (output["labels"] == 1)
                    boxes = output["boxes"][keep]
                    keypoints = output["keypoints"][keep]
                    counts.append(int(keep.sum().item()))
                    posed.append(
                        int(
                            sum(
                                int((person[:, 2] >= 1).sum().item()) >= 8
                                for person in keypoints
                            )
                        )
                    )
                    for box in boxes:
                        x1, y1, x2, y2 = box.tolist()
                        areas.append(max(0.0, (x2 - x1) * (y2 - y1)))
                    detections.append(
                        {
                            "boxes": boxes.detach().cpu().numpy(),
                            "keypoints": keypoints.detach().cpu().numpy(),
                        }
                    )
        result = {
            "person_count_mean": float(np.mean(counts)) if counts else 0.0,
            "person_present_fraction": float(np.mean(np.asarray(counts) > 0)) if counts else 0.0,
            "multiple_people_fraction": float(np.mean(np.asarray(counts) > 1)) if counts else 0.0,
            "posed_person_count_mean": float(np.mean(posed)) if posed else 0.0,
            "person_box_area_mean": float(np.mean(areas)) if areas else 0.0,
        }
        result.update(relational_pose_features(detections, frames))
        return result


def relational_pose_features(
    detections: list[dict[str, Any]],
    frames: list[np.ndarray],
) -> dict[str, float]:
    """Summarize inspectable multi-person geometry and sparse-frame motion."""
    count_values: list[int] = []
    largest_area_fractions: list[float] = []
    second_to_first_area_ratios: list[float] = []
    nearest_pair_distances: list[float] = []
    pair_ious: list[float] = []
    wrist_near_other: list[bool] = []
    center_motions: list[float] = []
    stable_single_transitions: list[bool] = []
    normalized_boxes_by_frame: list[np.ndarray] = []

    for detection, frame in zip(detections, frames):
        height, width = frame.shape[:2]
        scale = np.asarray([width, height, width, height], dtype=float)
        boxes = np.asarray(detection["boxes"], dtype=float).reshape(-1, 4)
        boxes = boxes / scale if len(boxes) else boxes
        normalized_boxes_by_frame.append(boxes)
        count_values.append(len(boxes))
        if not len(boxes):
            continue
        widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        areas = widths * heights
        sorted_areas = np.sort(areas)[::-1]
        largest_area_fractions.append(float(sorted_areas[0]))
        if len(sorted_areas) >= 2 and sorted_areas[0] > 0:
            second_to_first_area_ratios.append(
                float(sorted_areas[1] / sorted_areas[0])
            )
        if len(boxes) < 2:
            continue
        centers = np.column_stack(
            ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2)
        )
        frame_distances: list[float] = []
        frame_ious: list[float] = []
        for left in range(len(boxes)):
            for right in range(left + 1, len(boxes)):
                frame_distances.append(
                    float(np.linalg.norm(centers[left] - centers[right]))
                )
                intersection_left = np.maximum(
                    boxes[left, :2], boxes[right, :2]
                )
                intersection_right = np.minimum(
                    boxes[left, 2:], boxes[right, 2:]
                )
                intersection_size = np.maximum(
                    0.0, intersection_right - intersection_left
                )
                intersection = float(np.prod(intersection_size))
                union = float(areas[left] + areas[right] - intersection)
                frame_ious.append(intersection / union if union > 0 else 0.0)
        nearest_pair_distances.append(min(frame_distances))
        pair_ious.append(max(frame_ious))

        keypoints = np.asarray(detection["keypoints"], dtype=float)
        for person_index, person in enumerate(keypoints):
            for wrist_index in (9, 10):
                if person[wrist_index, 2] < 1:
                    continue
                wrist = person[wrist_index, :2] / np.asarray(
                    [width, height], dtype=float
                )
                near_another = False
                for other_index, other_box in enumerate(boxes):
                    if other_index == person_index:
                        continue
                    margin = 0.05
                    if (
                        other_box[0] - margin <= wrist[0] <= other_box[2] + margin
                        and other_box[1] - margin
                        <= wrist[1]
                        <= other_box[3] + margin
                    ):
                        near_another = True
                        break
                wrist_near_other.append(near_another)

    for previous, current in zip(
        normalized_boxes_by_frame, normalized_boxes_by_frame[1:]
    ):
        if not len(previous) or not len(current):
            continue
        previous_centers = np.column_stack(
            (
                (previous[:, 0] + previous[:, 2]) / 2,
                (previous[:, 1] + previous[:, 3]) / 2,
            )
        )
        current_centers = np.column_stack(
            (
                (current[:, 0] + current[:, 2]) / 2,
                (current[:, 1] + current[:, 3]) / 2,
            )
        )
        distances = np.linalg.norm(
            previous_centers[:, None, :] - current_centers[None, :, :],
            axis=2,
        )
        nearest_motion = float(
            np.mean(
                np.concatenate(
                    (distances.min(axis=0), distances.min(axis=1))
                )
            )
        )
        center_motions.append(nearest_motion)
        if len(previous) == 1 and len(current) == 1:
            stable_single_transitions.append(nearest_motion <= 0.08)

    return {
        "person_count_std": (
            float(np.std(count_values)) if count_values else 0.0
        ),
        "largest_person_area_fraction_mean": (
            float(np.mean(largest_area_fractions))
            if largest_area_fractions
            else 0.0
        ),
        "second_to_first_area_ratio_mean": (
            float(np.mean(second_to_first_area_ratios))
            if second_to_first_area_ratios
            else 0.0
        ),
        "nearest_pair_distance_mean": (
            float(np.mean(nearest_pair_distances))
            if nearest_pair_distances
            else 1.0
        ),
        "close_pair_fraction": (
            float(np.mean(np.asarray(nearest_pair_distances) <= 0.35))
            if nearest_pair_distances
            else 0.0
        ),
        "pair_iou_mean": float(np.mean(pair_ious)) if pair_ious else 0.0,
        "wrist_near_other_person_fraction": (
            float(np.mean(wrist_near_other)) if wrist_near_other else 0.0
        ),
        "person_center_motion_mean": (
            float(np.mean(center_motions)) if center_motions else 0.0
        ),
        "stable_single_person_transition_fraction": (
            float(np.mean(stable_single_transitions))
            if stable_single_transitions
            else 0.0
        ),
    }


class ClipScorer:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model_name).eval().to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

    def __call__(self, frames: list[np.ndarray]) -> dict:
        images = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
        inputs = self.processor(
            text=CLIP_PROMPTS,
            images=images,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            probabilities = self.model(**inputs).logits_per_image.softmax(dim=1).cpu().numpy()
        return {
            "prompts": CLIP_PROMPTS,
            "mean_probabilities": probabilities.mean(axis=0).tolist(),
            "max_probabilities": probabilities.max(axis=0).tolist(),
        }


class XClipScorer:
    def __init__(self, model_name: str, device: str):
        import torch
        from transformers import XCLIPModel, XCLIPProcessor

        self.torch = torch
        self.device = device
        self.model = XCLIPModel.from_pretrained(model_name).eval().to(device)
        self.processor = XCLIPProcessor.from_pretrained(model_name)

    def __call__(self, frames: list[np.ndarray]) -> dict:
        expected_frames = int(self.model.config.vision_config.num_frames)
        frames = uniformly_subsample_frames(frames, expected_frames)
        video = [
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for frame in frames
        ]
        inputs = xclip_processor_inputs(
            self.processor,
            video,
            text=XCLIP_PROMPTS,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            probabilities = (
                self.model(**inputs)
                .logits_per_video.softmax(dim=1)
                .cpu()
                .numpy()[0]
            )
        return {
            "prompts": XCLIP_PROMPTS,
            "probabilities": probabilities.tolist(),
            "sampled_frames": len(frames),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keypoints", action="store_true")
    parser.add_argument("--clip", action="store_true")
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--xclip", action="store_true")
    parser.add_argument("--xclip-model", default="microsoft/xclip-base-patch32")
    parser.add_argument(
        "--torch-threads",
        type=int,
        help="Bound CPU Torch intra-op threads for shared-server scoring.",
    )
    parser.add_argument(
        "--torch-home",
        type=Path,
        help="Readable Torch model cache for non-login/AFS-restricted shells.",
    )
    parser.add_argument(
        "--hf-home",
        type=Path,
        help="Readable Hugging Face cache for non-login/AFS-restricted shells.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--completed-from",
        type=Path,
        action="append",
        default=[],
        help="Additional append-only score file whose successful items are skipped.",
    )
    args = parser.parse_args()
    if args.num_shards < 1:
        raise SystemExit("num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")
    if args.torch_threads is not None and args.torch_threads < 1:
        raise SystemExit("torch-threads must be positive")
    if args.torch_home is not None:
        os.environ["TORCH_HOME"] = str(args.torch_home)
    if args.hf_home is not None:
        os.environ["HF_HOME"] = str(args.hf_home)
        os.environ["TRANSFORMERS_CACHE"] = str(args.hf_home)
    if args.torch_threads is not None:
        os.environ["OMP_NUM_THREADS"] = str(args.torch_threads)
        os.environ["MKL_NUM_THREADS"] = str(args.torch_threads)
        import torch

        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)

    rows = load_jsonl(args.manifest)
    rows = [
        row
        for row in rows
        if shard_for_item(row["item_id"], args.num_shards) == args.shard_index
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    required_sections = ["low_level"]
    if args.keypoints:
        required_sections.append("keypoints")
    if args.clip:
        required_sections.append("clip_scores")
    if args.xclip:
        required_sections.append("xclip_scores")
    completed = completed_item_ids(
        [args.out, *args.completed_from],
        tuple(required_sections),
    )
    keypoints = KeypointScorer(args.device) if args.keypoints else None
    clip = ClipScorer(args.clip_model, args.device) if args.clip else None
    xclip = XClipScorer(args.xclip_model, args.device) if args.xclip else None
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("a") as handle:
        for index, row in enumerate(rows, 1):
            if row["item_id"] in completed:
                continue
            path = resolve_clip(row, args.manifest)
            error = None
            if not path.is_file():
                frames, media = [], {}
                error = f"FileNotFoundError: {path}"
            else:
                try:
                    frames, media = sample_frames(
                        path,
                        args.frames,
                        row.get("media_start_sec"),
                        row.get("media_end_sec"),
                    )
                    if not frames:
                        error = "ValueError: no decodable frames"
                except (OSError, TypeError, ValueError) as exc:
                    frames, media = [], {}
                    error = f"{type(exc).__name__}: {exc}"
            result = {
                "item_id": row["item_id"],
                "pillar": row.get("pillar") or row["source_modality"],
                "uid": row["uid"],
                "gold_scene_visible": row.get("gold_scene_visible"),
                "gold_usable": row.get("gold_usable"),
                "clip": str(path),
                "media": media,
                "low_level": low_level_features(frames),
                "keypoints": keypoints(frames) if keypoints and frames else None,
                "clip_scores": clip(frames) if clip and frames else None,
                "xclip_scores": xclip(frames) if xclip and frames else None,
                "error": error,
            }
            handle.write(json.dumps(result, sort_keys=True) + "\n")
            handle.flush()
            if index % 100 == 0 or index == len(rows):
                print(
                    f"{index}/{len(rows)} {row['item_id']} "
                    f"{'ok' if error is None else error}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
