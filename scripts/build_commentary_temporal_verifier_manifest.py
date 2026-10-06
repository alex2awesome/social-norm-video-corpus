#!/usr/bin/env python3
"""Render bounded motion storyboards for label-blind temporal verification.

The renderer retains an unmasked human-audit copy and creates an OCR-masked VLM
copy. Persistent, dense overlay-text runs trigger narrow full-row redactions.
This removes complete news headlines while preserving scene evidence outside
the actual caption row. Use export_commentary_temporal_model_manifest.py before
copying a manifest to a model host.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import cv2
import numpy as np
import pytesseract


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resize_width(frame: np.ndarray, width: int) -> np.ndarray:
    """Resize one frame to a fixed width while preserving its aspect ratio."""
    if width <= 0:
        raise ValueError("cell width must be positive")
    height, current_width = frame.shape[:2]
    target_height = max(1, round(height * width / current_width))
    interpolation = cv2.INTER_AREA if width < current_width else cv2.INTER_CUBIC
    return cv2.resize(frame, (width, target_height), interpolation=interpolation)


def resize_bounded(
    frame: np.ndarray,
    maximum_width: int,
    maximum_height: int,
) -> np.ndarray:
    """Fit a frame inside a pixel budget without cropping or distorting it."""
    if maximum_width <= 0 or maximum_height <= 0:
        raise ValueError("maximum cell dimensions must be positive")
    height, width = frame.shape[:2]
    scale = min(maximum_width / width, maximum_height / height)
    target_width = max(1, round(width * scale))
    target_height = max(1, round(height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    return cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=interpolation,
    )


def bounded_sampling_fps(
    duration_seconds: float,
    requested_fps: float,
    maximum_frames: int | None,
) -> float:
    """Lower sampling rate for long windows while preserving the full span."""
    if duration_seconds <= 0 or requested_fps <= 0:
        raise ValueError("duration and requested fps must be positive")
    if maximum_frames is None:
        return requested_fps
    if maximum_frames <= 0:
        raise ValueError("maximum frames must be positive")
    return min(requested_fps, maximum_frames / duration_seconds)


def _valid_ocr_word(text: str, confidence: float, threshold: float) -> bool:
    alphanumeric = [character for character in text if character.isalnum()]
    return confidence >= threshold and (
        len(alphanumeric) >= 2 or any(character.isdigit() for character in text)
    )


def _clamp_box(
    box: tuple[int, int, int, int],
    width: int,
    height: int,
    x_padding: int,
    y_padding: int,
) -> tuple[int, int, int, int]:
    x, y, w, h = box
    x1 = max(0, x - x_padding)
    y1 = max(0, y - y_padding)
    x2 = min(width, x + w + x_padding)
    y2 = min(height, y + h + y_padding)
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def is_overlay_band_box(
    box: tuple[int, int, int, int],
    frame_height: int,
    top_fraction: float = 0.20,
    bottom_fraction: float = 0.72,
) -> bool:
    """Keep OCR boxes in common title/lower-third regions, not scene interiors."""
    _, y, _, height = box
    return y < frame_height * top_fraction or y + height > frame_height * bottom_fraction


def uses_portrait_all_screen_ocr(
    frame_width: int,
    frame_height: int,
    enabled: bool,
) -> bool:
    """Restrict aggressive interior-caption OCR to tall social-video layouts."""
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    return enabled and frame_width / frame_height < 0.80


def _overlap_over_smaller(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> float:
    lx, ly, lw, lh = left
    rx, ry, rw, rh = right
    intersection_width = max(0, min(lx + lw, rx + rw) - max(lx, rx))
    intersection_height = max(0, min(ly + lh, ry + rh) - max(ly, ry))
    intersection = intersection_width * intersection_height
    smaller = min(lw * lh, rw * rh)
    return intersection / smaller if smaller else 0.0


def merge_overlapping_boxes(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Merge duplicate OCR boxes produced by multiple page-segmentation modes."""
    merged: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes):
        for position, existing in enumerate(merged):
            if _overlap_over_smaller(box, existing) < 0.65:
                continue
            x1 = min(box[0], existing[0])
            y1 = min(box[1], existing[1])
            x2 = max(box[0] + box[2], existing[0] + existing[2])
            y2 = max(box[1] + box[3], existing[1] + existing[3])
            merged[position] = (x1, y1, x2 - x1, y2 - y1)
            break
        else:
            merged.append(box)
    return merged


def persistent_boxes(
    boxes_by_frame: list[list[tuple[int, int, int, int]]],
    minimum_frames: int = 2,
) -> list[tuple[int, int, int, int]]:
    """Keep position-consistent OCR regions observed in multiple sampled frames."""
    if minimum_frames <= 0:
        raise ValueError("minimum OCR persistence must be positive")
    clusters: list[dict[str, Any]] = []
    for frame_number, boxes in enumerate(boxes_by_frame):
        for box in boxes:
            candidates = sorted(
                (
                    (_overlap_over_smaller(box, cluster["anchor"]), cluster)
                    for cluster in clusters
                    if frame_number not in cluster["frames"]
                ),
                key=lambda value: value[0],
                reverse=True,
            )
            for overlap, cluster in candidates:
                if overlap < 0.65:
                    continue
                cluster["boxes"].append(box)
                cluster["frames"].add(frame_number)
                break
            else:
                clusters.append(
                    {
                        "anchor": box,
                        "boxes": [box],
                        "frames": {frame_number},
                    }
                )
    persistent = []
    for cluster in clusters:
        if len(cluster["frames"]) < minimum_frames:
            continue
        x1 = min(box[0] for box in cluster["boxes"])
        y1 = min(box[1] for box in cluster["boxes"])
        x2 = max(box[0] + box[2] for box in cluster["boxes"])
        y2 = max(box[1] + box[3] for box in cluster["boxes"])
        persistent.append((x1, y1, x2 - x1, y2 - y1))
    return persistent


def select_overlay_text_boxes(
    boxes: list[tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int, int, int]]:
    """Select dense screen-aligned word runs and reject isolated scene signage."""
    eligible = [
        box
        for box in boxes
        if box[3] <= frame_height * 0.20 and box[2] <= frame_width * 0.90
    ]
    line_groups: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(eligible, key=lambda value: value[1] + value[3] / 2):
        center = box[1] + box[3] / 2
        for group in line_groups:
            group_centers = [value[1] + value[3] / 2 for value in group]
            group_heights = [value[3] for value in group]
            tolerance = max(4.0, 0.45 * float(np.median(group_heights + [box[3]])))
            if abs(center - float(np.median(group_centers))) <= tolerance:
                group.append(box)
                break
        else:
            line_groups.append([box])

    selected_lines = []
    for group in line_groups:
        unique = []
        for box in sorted(group):
            if any(_overlap_over_smaller(box, existing) >= 0.75 for existing in unique):
                continue
            unique.append(box)
        if not unique:
            continue
        typical_height = float(np.median([box[3] for box in unique]))
        maximum_gap = max(12.0, 1.5 * typical_height)
        runs: list[list[tuple[int, int, int, int]]] = []
        for box in sorted(unique, key=lambda value: value[0]):
            if not runs:
                runs.append([box])
                continue
            previous = runs[-1][-1]
            gap = box[0] - (previous[0] + previous[2])
            if gap <= maximum_gap:
                runs[-1].append(box)
            else:
                runs.append([box])
        for run in runs:
            total_width = sum(box[2] for box in run)
            span = max(box[0] + box[2] for box in run) - min(
                box[0] for box in run
            )
            if len(run) >= 3 and (
                total_width >= frame_width * 0.14
                or span >= frame_width * 0.22
            ):
                vertical_padding = max(
                    4,
                    round(0.25 * float(np.median([box[3] for box in run]))),
                )
                y1 = max(0, min(box[1] for box in run) - vertical_padding)
                y2 = min(
                    frame_height,
                    max(box[1] + box[3] for box in run) + vertical_padding,
                )
                selected_lines.append((0, y1, frame_width, y2 - y1))
            elif len(run) == 2 and total_width >= frame_width * 0.22:
                vertical_padding = max(
                    4,
                    round(0.25 * float(np.median([box[3] for box in run]))),
                )
                y1 = max(0, min(box[1] for box in run) - vertical_padding)
                y2 = min(
                    frame_height,
                    max(box[1] + box[3] for box in run) + vertical_padding,
                )
                selected_lines.append((0, y1, frame_width, y2 - y1))
            elif len(run) == 1 and total_width >= frame_width * 0.45:
                vertical_padding = max(4, round(0.25 * run[0][3]))
                y1 = max(0, run[0][1] - vertical_padding)
                y2 = min(
                    frame_height,
                    run[0][1] + run[0][3] + vertical_padding,
                )
                selected_lines.append((0, y1, frame_width, y2 - y1))
    return merge_overlapping_boxes(selected_lines)


def ocr_word_boxes(
    frame: np.ndarray,
    confidence_threshold: float,
    ocr_scale: float,
    psm_modes: tuple[int, ...] = (11,),
    overlay_bands_only: bool = True,
) -> list[tuple[int, int, int, int]]:
    """Detect high-confidence word boxes without forcing scene pixels into text."""
    if ocr_scale <= 0:
        raise ValueError("OCR scale must be positive")
    enlarged = cv2.resize(
        frame,
        None,
        fx=ocr_scale,
        fy=ocr_scale,
        interpolation=cv2.INTER_CUBIC,
    )
    rgb = cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)
    frame_height, frame_width = frame.shape[:2]
    collected: list[tuple[int, int, int, int]] = []
    for psm in psm_modes:
        data = pytesseract.image_to_data(
            rgb,
            output_type=pytesseract.Output.DICT,
            config=f"--psm {psm}",
        )
        for position, text in enumerate(data["text"]):
            try:
                confidence = float(data["conf"][position])
            except (TypeError, ValueError):
                continue
            if not _valid_ocr_word(text, confidence, confidence_threshold):
                continue
            scaled = (
                round(int(data["left"][position]) / ocr_scale),
                round(int(data["top"][position]) / ocr_scale),
                max(1, round(int(data["width"][position]) / ocr_scale)),
                max(1, round(int(data["height"][position]) / ocr_scale)),
            )
            if overlay_bands_only and not is_overlay_band_box(
                scaled,
                frame_height,
            ):
                continue
            if scaled[3] > frame_height * 0.14 or scaled[2] > frame_width * 0.65:
                continue
            collected.append(
                _clamp_box(
                    scaled,
                    frame_width,
                    frame_height,
                    x_padding=max(3, round(scaled[3] * 0.18)),
                    y_padding=max(2, round(scaled[3] * 0.22)),
                )
            )
    return collected


def easyocr_detect_boxes(
    frame: np.ndarray,
    reader: Any,
) -> list[tuple[int, int, int, int]]:
    """Run CRAFT text detection without using recognized words as evidence."""
    horizontal_by_image, free_by_image = reader.detect(
        frame,
        text_threshold=0.4,
        low_text=0.3,
        link_threshold=0.3,
        canvas_size=1280,
        mag_ratio=2,
    )
    if free_by_image and free_by_image[0]:
        raise ValueError("rotated EasyOCR text boxes are not supported")
    height, width = frame.shape[:2]
    boxes = []
    for raw in horizontal_by_image[0] if horizontal_by_image else []:
        x1, x2, y1, y2 = (int(round(value)) for value in raw)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append(
            _clamp_box(
                (x1, y1, x2 - x1, y2 - y1),
                width,
                height,
                x_padding=max(3, round((y2 - y1) * 0.18)),
                y_padding=max(2, round((y2 - y1) * 0.22)),
            )
        )
    return boxes


def select_easyocr_caption_rows(
    boxes: list[tuple[int, int, int, int]],
    frame_width: int,
    frame_height: int,
) -> list[tuple[int, int, int, int]]:
    """Convert persistent CRAFT detections into complete neutral text rows."""
    rows = []
    for x, y, width, height in boxes:
        if height > frame_height * 0.30 or width < frame_width * 0.08:
            continue
        vertical_padding = max(4, round(0.20 * height))
        y1 = max(0, y - vertical_padding)
        y2 = min(frame_height, y + height + vertical_padding)
        rows.append((0, y1, frame_width, y2 - y1))
    return merge_overlapping_boxes(rows)


def mask_text_boxes(
    frame: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    """Cover OCR boxes with a neutral flat patch that contains no readable text."""
    masked = frame.copy()
    for x, y, width, height in boxes:
        crop = frame[y : y + height, x : x + width]
        if crop.size:
            color = tuple(int(value) for value in np.median(crop, axis=(0, 1)))
        else:
            color = (96, 96, 96)
        cv2.rectangle(
            masked,
            (x, y),
            (x + width - 1, y + height - 1),
            color,
            thickness=-1,
        )
    return masked


def tile_frames(
    frames: list[np.ndarray],
    columns: int,
    rows: int,
    padding: int = 4,
    margin: int = 4,
) -> list[np.ndarray]:
    """Tile equal-sized chronological frames into one or more page images."""
    if not frames:
        raise ValueError("cannot tile an empty frame list")
    if columns <= 0 or rows <= 0:
        raise ValueError("grid dimensions must be positive")
    height, width = frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in frames):
        raise ValueError("all frames must have the same size")
    page_capacity = columns * rows
    pages = []
    for offset in range(0, len(frames), page_capacity):
        page_frames = frames[offset : offset + page_capacity]
        canvas_height = 2 * margin + rows * height + (rows - 1) * padding
        canvas_width = 2 * margin + columns * width + (columns - 1) * padding
        canvas = np.full((canvas_height, canvas_width, 3), 32, dtype=np.uint8)
        for position, frame in enumerate(page_frames):
            row, column = divmod(position, columns)
            x = margin + column * (width + padding)
            y = margin + row * (height + padding)
            canvas[y : y + height, x : x + width] = frame
        pages.append(canvas)
    return pages


def write_pages(
    pages: list[np.ndarray],
    directory: Path,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for number, page in enumerate(pages, 1):
        path = directory / f"page_{number:02d}.jpg"
        if not cv2.imwrite(str(path), page, [cv2.IMWRITE_JPEG_QUALITY, 92]):
            raise RuntimeError(f"failed to write {path}")
        paths.append(path)
    return paths


def portable_path(path: Path, manifest_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(manifest_dir.resolve()))
    except ValueError:
        return str(path.resolve())


def build_rows(
    windows: list[dict[str, Any]],
    semantic_rows: list[dict[str, Any]],
    proxy_dir: Path,
    output_dir: Path,
    manifest_dir: Path,
    ffmpeg: str,
    fps: float = 4.0,
    columns: int = 8,
    rows_per_page: int = 4,
    cell_width: int = 300,
    ocr_mask: bool = False,
    ocr_confidence: float = 50.0,
    ocr_scale: float = 2.0,
    ocr_sample_every: int = 4,
    ocr_min_sampled_frames: int = 2,
    cell_max_height: int | None = None,
    ocr_all_screen: bool = False,
    portrait_text_detector: Any | None = None,
    max_frames_per_window: int | None = None,
) -> list[dict[str, Any]]:
    semantic = {int(row["audit_index"]): row for row in semantic_rows}
    seen: set[int] = set()
    rows = []
    for spec in windows:
        index = int(spec["audit_index"])
        if index in seen:
            raise ValueError(f"duplicate audit_index {index}")
        seen.add(index)
        if index not in semantic:
            raise ValueError(f"missing semantic row {index}")
        start = float(spec["start_sec"])
        end = float(spec["end_sec"])
        if start < 0 or end <= start:
            raise ValueError(f"invalid window for {index}: {start}, {end}")
        row_fps = bounded_sampling_fps(
            end - start,
            fps,
            max_frames_per_window,
        )

        candidate_id = semantic[index]["candidate_id"]
        proxy = proxy_dir / f"{candidate_id}.mp4"
        if not proxy.is_file():
            raise FileNotFoundError(proxy)
        with tempfile.TemporaryDirectory(prefix=f"temporal-{index:04d}-") as temp:
            frame_pattern = Path(temp) / "frame_%06d.png"
            command = [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                str(start),
                "-t",
                str(end - start),
                "-i",
                str(proxy),
                "-vf",
                f"fps={row_fps}",
                "-vsync",
                "vfr",
                str(frame_pattern),
            ]
            subprocess.run(command, check=True)
            frame_paths = sorted(Path(temp).glob("frame_*.png"))
            if not frame_paths:
                raise RuntimeError(f"no frames rendered for {index}")
            unmasked_frames = []
            for frame_path in frame_paths:
                source_frame = cv2.imread(str(frame_path))
                if source_frame is None:
                    raise RuntimeError(f"failed to read {frame_path}")
                frame = (
                    resize_bounded(source_frame, cell_width, cell_max_height)
                    if cell_max_height is not None
                    else resize_width(source_frame, cell_width)
                )
                unmasked_frames.append(frame)

        if ocr_sample_every <= 0:
            raise ValueError("OCR sample stride must be positive")
        sample_positions = list(range(0, len(unmasked_frames), ocr_sample_every))
        if sample_positions[-1] != len(unmasked_frames) - 1:
            sample_positions.append(len(unmasked_frames) - 1)
        sampled_boxes = []
        sampled_easyocr_boxes = []
        if ocr_mask:
            portrait_all_screen = uses_portrait_all_screen_ocr(
                unmasked_frames[0].shape[1],
                unmasked_frames[0].shape[0],
                ocr_all_screen,
            )
            for position in sample_positions:
                boxes = ocr_word_boxes(
                    unmasked_frames[position],
                    ocr_confidence,
                    ocr_scale,
                    psm_modes=(11,),
                    overlay_bands_only=not portrait_all_screen,
                )
                if portrait_all_screen and portrait_text_detector is not None:
                    sampled_easyocr_boxes.append(
                        easyocr_detect_boxes(
                            unmasked_frames[position],
                            portrait_text_detector,
                        )
                    )
                else:
                    sampled_easyocr_boxes.append([])
                sampled_boxes.append(boxes)
        else:
            portrait_all_screen = False
        persistent_candidate_boxes = persistent_boxes(
            sampled_boxes,
            minimum_frames=ocr_min_sampled_frames,
        ) if ocr_mask else []
        persistent_easyocr_boxes = persistent_boxes(
            sampled_easyocr_boxes,
            minimum_frames=ocr_min_sampled_frames,
        ) if ocr_mask else []
        selected_overlay_boxes = select_overlay_text_boxes(
            persistent_candidate_boxes,
            unmasked_frames[0].shape[1],
            unmasked_frames[0].shape[0],
        ) if ocr_mask else []
        if ocr_mask and persistent_easyocr_boxes:
            selected_overlay_boxes = merge_overlapping_boxes(
                selected_overlay_boxes
                + select_easyocr_caption_rows(
                    persistent_easyocr_boxes,
                    unmasked_frames[0].shape[1],
                    unmasked_frames[0].shape[0],
                )
            )
        masked_frames = [
            mask_text_boxes(frame, selected_overlay_boxes)
            for frame in unmasked_frames
        ]

        unmasked_pages = write_pages(
            tile_frames(unmasked_frames, columns, rows_per_page),
            output_dir / "pages_unmasked_human_only" / f"{index:04d}",
        )
        if ocr_mask:
            pages = write_pages(
                tile_frames(masked_frames, columns, rows_per_page),
                output_dir / "pages_ocr_masked" / f"{index:04d}",
            )
        else:
            pages = unmasked_pages
        portable_paths = [portable_path(page, manifest_dir) for page in pages]
        unmasked_portable_paths = [
            portable_path(page, manifest_dir) for page in unmasked_pages
        ]
        rows.append(
            {
                "audit_index": index,
                "candidate_id": candidate_id,
                "uid": semantic[index]["uid"],
                "fallible_claim": semantic[index]["norm"],
                "window_start_sec": start,
                "window_end_sec": end,
                "frame_order": (
                    f"Pages are chronological; within each {columns}x"
                    f"{rows_per_page} page read left-to-right then "
                    f"top-to-bottom at {row_fps:g} fps."
                ),
                "fps": row_fps,
                "requested_fps": fps,
                "max_frames_per_window": max_frames_per_window,
                "grid_columns": columns,
                "grid_rows": rows_per_page,
                "cell_width": unmasked_frames[0].shape[1],
                "cell_height": unmasked_frames[0].shape[0],
                "cell_max_width": cell_width,
                "cell_max_height": cell_max_height,
                "ocr_masked": ocr_mask,
                "ocr_mask_method": (
                    (
                        "tesseract_psm11_plus_easyocr_craft_persistent_dense_"
                        "portrait_all_screen_full_row_redaction"
                        if portrait_all_screen
                        else
                        "tesseract_psm11_persistent_dense_overlay_"
                        "full_row_redaction"
                    )
                    if ocr_mask
                    else None
                ),
                "ocr_all_screen": ocr_all_screen if ocr_mask else None,
                "ocr_portrait_all_screen_applied": (
                    portrait_all_screen if ocr_mask else None
                ),
                "ocr_portrait_easyocr_enabled": (
                    portrait_text_detector is not None if ocr_mask else None
                ),
                "ocr_confidence_threshold": ocr_confidence if ocr_mask else None,
                "ocr_sample_every_n_frames": (
                    ocr_sample_every if ocr_mask else None
                ),
                "ocr_min_sampled_frames": (
                    ocr_min_sampled_frames if ocr_mask else None
                ),
                "ocr_sampled_frames": (
                    len(sample_positions) if ocr_mask else 0
                ),
                "ocr_sampled_frames_with_boxes": (
                    sum(bool(boxes) for boxes in sampled_boxes)
                    if ocr_mask
                    else 0
                ),
                "ocr_persistent_candidate_boxes_total": len(
                    persistent_candidate_boxes
                ),
                "ocr_persistent_easyocr_boxes_total": len(
                    persistent_easyocr_boxes
                ),
                "ocr_selected_overlay_boxes_total": len(
                    selected_overlay_boxes
                ),
                "source_frame_count": len(unmasked_frames),
                "page_paths": portable_paths,
                "page_sha256": [sha256(page) for page in pages],
                "human_unmasked_page_paths": unmasked_portable_paths,
                "human_unmasked_page_sha256": [
                    sha256(page) for page in unmasked_pages
                ],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--windows", type=Path, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--proxy-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--fps", type=float, default=4.0)
    parser.add_argument(
        "--max-frames-per-window",
        type=int,
        help=(
            "adaptively lower FPS for long windows so each row stays within "
            "a bounded multimodal token budget"
        ),
    )
    parser.add_argument("--columns", type=int, default=8)
    parser.add_argument("--rows-per-page", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=300)
    parser.add_argument(
        "--cell-max-height",
        type=int,
        help=(
            "optional maximum cell height; with --cell-width this creates an "
            "aspect-preserving pixel budget for portrait sources"
        ),
    )
    parser.add_argument("--ocr-mask", action="store_true")
    parser.add_argument(
        "--ocr-all-screen",
        action="store_true",
        help=(
            "allow persistent dense text runs anywhere in the frame to trigger "
            "redaction; needed for centered social-video headlines"
        ),
    )
    parser.add_argument(
        "--portrait-easyocr",
        action="store_true",
        help=(
            "supplement Tesseract with EasyOCR CRAFT detection for bounded "
            "portrait sources; weights must already exist locally"
        ),
    )
    parser.add_argument(
        "--easyocr-model-storage-directory",
        type=Path,
        help="optional directory containing pre-downloaded EasyOCR weights",
    )
    parser.add_argument("--ocr-confidence", type=float, default=50.0)
    parser.add_argument("--ocr-scale", type=float, default=2.0)
    parser.add_argument(
        "--ocr-sample-every",
        type=int,
        default=4,
        help=(
            "run OCR on every Nth temporal frame, union the boxes, and redact "
            "those regions from every frame"
        ),
    )
    parser.add_argument(
        "--ocr-min-sampled-frames",
        type=int,
        default=2,
        help="require an overlapping OCR region in at least this many samples",
    )
    parser.add_argument(
        "--tesseract-cmd",
        help="optional explicit path to the tesseract executable",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd
    portrait_text_detector = None
    if args.portrait_easyocr:
        import easyocr

        reader_options: dict[str, Any] = {
            "gpu": False,
            "download_enabled": False,
            "verbose": False,
        }
        if args.easyocr_model_storage_directory is not None:
            reader_options["model_storage_directory"] = str(
                args.easyocr_model_storage_directory
            )
        portrait_text_detector = easyocr.Reader(["en"], **reader_options)
    rows = build_rows(
        read_jsonl(args.windows),
        read_jsonl(args.semantic),
        args.proxy_dir,
        args.output_dir,
        args.out.parent,
        args.ffmpeg,
        args.fps,
        args.columns,
        args.rows_per_page,
        args.cell_width,
        args.ocr_mask,
        args.ocr_confidence,
        args.ocr_scale,
        args.ocr_sample_every,
        args.ocr_min_sampled_frames,
        args.cell_max_height,
        args.ocr_all_screen,
        portrait_text_detector,
        args.max_frames_per_window,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    print(json.dumps({"rows": len(rows), "pages": sum(len(r["page_paths"]) for r in rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
