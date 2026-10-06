#!/usr/bin/env python3
"""Score localized witnessed candidates with an acoustic speaker proxy.

This is deliberately not called diarization.  ECAPA embeddings estimate voice
similarity around a transcript-localized candidate; they do not identify a
person or prove a bystander role.  Outputs are shadow-only features.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
from typing import Any

import numpy as np


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denom) if denom else math.nan


def surrounding_segments(
    segments: list[dict[str, Any]],
    candidate_start: float,
    candidate_end: float,
    *,
    lookaround_sec: float = 8.0,
    max_each_side: int = 4,
    minimum_duration: float = 0.35,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid = [
        row
        for row in segments
        if isinstance(row.get("clip_start"), (int, float))
        and isinstance(row.get("clip_end"), (int, float))
        and float(row["clip_end"]) - max(0.0, float(row["clip_start"]))
        >= minimum_duration
    ]
    before = [
        row
        for row in valid
        if float(row["clip_end"]) <= candidate_start
        and candidate_start - float(row["clip_end"]) <= lookaround_sec
    ][-max_each_side:]
    after = [
        row
        for row in valid
        if float(row["clip_start"]) >= candidate_end
        and float(row["clip_start"]) - candidate_end <= lookaround_sec
    ][:max_each_side]
    return before, after


def summarize_voice_proxy(
    candidate: np.ndarray,
    before: list[np.ndarray],
    after: list[np.ndarray],
    *,
    same_voice_threshold: float = 0.65,
    distinct_voice_threshold: float = 0.55,
) -> dict[str, float]:
    before_similarity = [cosine_similarity(candidate, row) for row in before]
    after_similarity = [cosine_similarity(candidate, row) for row in after]
    immediate_before = before_similarity[-1] if before_similarity else math.nan
    immediate_after = after_similarity[0] if after_similarity else math.nan
    prior_next_similarity = (
        cosine_similarity(before[-1], after[0]) if before and after else math.nan
    )
    finite_prior = [value for value in before_similarity if math.isfinite(value)]
    distinct_prior = bool(finite_prior) and max(finite_prior) < distinct_voice_threshold
    sandwich = (
        math.isfinite(immediate_before)
        and math.isfinite(immediate_after)
        and math.isfinite(prior_next_similarity)
        and immediate_before < distinct_voice_threshold
        and immediate_after < distinct_voice_threshold
        and prior_next_similarity >= same_voice_threshold
    )
    return {
        "speaker_proxy.available": 1.0,
        "speaker_proxy.before_segments": float(len(before)),
        "speaker_proxy.after_segments": float(len(after)),
        "speaker_proxy.immediate_before_similarity": immediate_before,
        "speaker_proxy.immediate_after_similarity": immediate_after,
        "speaker_proxy.max_prior_similarity": max(finite_prior) if finite_prior else math.nan,
        "speaker_proxy.prior_next_similarity": prior_next_similarity,
        "speaker_proxy.differs_from_immediate_before": float(
            math.isfinite(immediate_before)
            and immediate_before < distinct_voice_threshold
        ),
        "speaker_proxy.new_vs_recent_prior": float(distinct_prior),
        "speaker_proxy.sandwiched_third_voice": float(sandwich),
    }


def decode_audio(path: Path, sample_rate: int = 16000) -> np.ndarray:
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return np.frombuffer(completed.stdout, dtype="<i2").astype(np.float32) / 32768.0


def audio_window(
    waveform: np.ndarray,
    start: float,
    end: float,
    sample_rate: int,
    *,
    minimum_seconds: float = 1.0,
) -> np.ndarray:
    start = max(0.0, start)
    end = max(start, end)
    values = waveform[int(start * sample_rate) : int(end * sample_rate)]
    minimum = int(minimum_seconds * sample_rate)
    if len(values) < minimum:
        values = np.pad(values, (0, minimum - len(values)))
    return values.astype(np.float32, copy=False)


def encode_windows(model: Any, windows: list[np.ndarray]) -> list[np.ndarray]:
    import torch

    lengths = [len(row) for row in windows]
    maximum = max(lengths)
    batch = torch.stack(
        [
            torch.nn.functional.pad(torch.from_numpy(row.copy()), (0, maximum - len(row)))
            for row in windows
        ]
    )
    relative = torch.tensor([length / maximum for length in lengths])
    with torch.inference_mode():
        encoded = model.encode_batch(batch, wav_lens=relative).squeeze(1)
        encoded = torch.nn.functional.normalize(encoded, dim=-1)
    return [row.detach().cpu().numpy() for row in encoded]


def load_model(source: str, cache: Path) -> Any:
    try:
        from speechbrain.inference.speaker import EncoderClassifier
    except ModuleNotFoundError:
        from speechbrain.pretrained import EncoderClassifier
    return EncoderClassifier.from_hparams(source=source, savedir=str(cache))


def score_candidate(
    candidate: dict[str, Any],
    benchmark: dict[str, Any],
    waveform: np.ndarray,
    model: Any,
    sample_rate: int,
) -> dict[str, Any]:
    start = float(candidate["candidate_start_sec"])
    end = float(candidate["candidate_end_sec"])
    before_rows, after_rows = surrounding_segments(
        benchmark.get("aligned_transcript") or [], start, end
    )
    intervals = [
        (start, end),
        *[(max(0.0, float(row["clip_start"])), float(row["clip_end"])) for row in before_rows],
        *[(max(0.0, float(row["clip_start"])), float(row["clip_end"])) for row in after_rows],
    ]
    windows = [audio_window(waveform, a, b, sample_rate) for a, b in intervals]
    encoded = encode_windows(model, windows)
    before = encoded[1 : 1 + len(before_rows)]
    after = encoded[1 + len(before_rows) :]
    return {
        "candidate_id": candidate["candidate_id"],
        "item_id": candidate["item_id"],
        "uid": candidate["uid"],
        "candidate_start_sec": start,
        "candidate_end_sec": end,
        "candidate_text": candidate["candidate_text"],
        "features": summarize_voice_proxy(encoded[0], before, after),
        "context_before": [str(row.get("text") or "") for row in before_rows],
        "context_after": [str(row.get("text") or "") for row in after_rows],
        "policy": "shadow_feature_only_not_diarization_or_acceptance",
        "error": None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("model_cache/speechbrain_spkrec_ecapa_voxceleb"))
    parser.add_argument("--model", default="speechbrain/spkrec-ecapa-voxceleb")
    args = parser.parse_args()
    candidates = read_jsonl(args.candidates)
    benchmark = {row["item_id"]: row for row in read_jsonl(args.benchmark)}
    if len(benchmark) != len(read_jsonl(args.benchmark)):
        raise ValueError("duplicate benchmark item_id")
    model = load_model(args.model, args.cache)
    by_media: dict[str, np.ndarray] = {}
    output = []
    for index, candidate in enumerate(candidates, 1):
        try:
            source = benchmark[candidate["item_id"]]
            media = str(source["source_clip"])
            if media not in by_media:
                by_media[media] = decode_audio(Path(media))
            row = score_candidate(candidate, source, by_media[media], model, 16000)
        except Exception as exc:
            row = {
                "candidate_id": candidate.get("candidate_id"),
                "item_id": candidate.get("item_id"),
                "uid": candidate.get("uid"),
                "features": None,
                "policy": "shadow_feature_only_not_diarization_or_acceptance",
                "error": f"{type(exc).__name__}: {exc}",
            }
        output.append(row)
        print(f"{index}/{len(candidates)} {row['candidate_id']} error={row['error']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
