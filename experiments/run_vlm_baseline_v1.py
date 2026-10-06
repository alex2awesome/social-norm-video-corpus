#!/usr/bin/env python3
"""Zero-shot VLM baseline on the R1 reaction-prediction manifest.

Samples frames from each pre-reaction clip, asks a vision-language model
ATOMIC questions (never "is this a norm?"), and appends validated JSON rows.
The contract includes ``reaction_already_visible`` — a clip-integrity audit
field: if the model sees the reaction inside a positive clip, our window
guard leaked.  Same dynamic-lifecycle discipline as the 70B text pass: load,
batch, exit; GPU choice and memory gating live in the launch wrapper.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

RUNNER_VERSION = "run_vlm_baseline_v1"
JSON_BLOCK = re.compile(r"\{.*\}", re.S)

ROLES = ("bystander", "victim", "authority", "camera_person", "nobody", "unclear")

SYSTEM_PROMPT = """You see frames sampled from a short video clip of an unfolding \
social situation. The clip ends BEFORE anyone visibly responds (if it was cut \
correctly). Answer from the frames only.

Return ONE JSON object, no prose:
{"people_visible": integer count of distinct people you can see,
 "reaction_already_visible": true if someone is ALREADY visibly reacting to \
another person's behavior in these frames (shouting at them, intervening, \
recoiling), else false,
 "will_react_probability": number 0.0-1.0 that someone audibly or visibly \
reacts to the central behavior within the next ten seconds,
 "predicted_responder": one of %s,
 "predicted_strength": integer 1-5 (1 = mild remark, 5 = visceral outburst) \
or null if no reaction expected,
 "behavior_seen": one short phrase describing the central behavior}""" % json.dumps(list(ROLES))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def extract_frames(ffmpeg: str, clip: Path, n_frames: int, tmp_dir: Path) -> list[bytes]:
    pattern = tmp_dir / "frame_%02d.jpg"
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
         "-i", str(clip),
         "-vf", f"thumbnail=8,scale='min(384,iw)':-2,fps=1",
         "-frames:v", str(n_frames), str(pattern)],
        capture_output=True,
    )
    frames = sorted(tmp_dir.glob("frame_*.jpg"))
    if not frames:  # fallback: plain uniform sampling
        subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
             "-i", str(clip), "-vf", "scale='min(384,iw)':-2,fps=1/2",
             "-frames:v", str(n_frames), str(pattern)],
            capture_output=True,
        )
        frames = sorted(tmp_dir.glob("frame_*.jpg"))
    payload = [f.read_bytes() for f in frames[:n_frames]]
    for f in frames:
        f.unlink(missing_ok=True)
    return payload


def frame_message(frames: list[bytes]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [
        {"type": "text",
         "text": f"{len(frames)} frames in temporal order. Answer the JSON contract."}
    ]
    for blob in frames:
        encoded = base64.b64encode(blob).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}})
    return content


def parse_result(text: str) -> dict[str, Any]:
    match = JSON_BLOCK.search(text)
    if not match:
        raise ValueError("no JSON in output")
    result = json.loads(match.group(0))
    p = result.get("will_react_probability")
    if not isinstance(p, (int, float)) or not 0.0 <= float(p) <= 1.0:
        raise ValueError(f"bad will_react_probability: {p!r}")
    if result.get("predicted_responder") not in ROLES:
        raise ValueError(f"bad predicted_responder: {result.get('predicted_responder')!r}")
    if result.get("reaction_already_visible") not in (True, False):
        raise ValueError("bad reaction_already_visible")
    strength = result.get("predicted_strength")
    if strength is not None and strength not in (1, 2, 3, 4, 5):
        raise ValueError(f"bad predicted_strength: {strength!r}")
    return result


def select_pilot(manifest: Path, size: int) -> list[dict[str, Any]]:
    """Deterministic balanced pilot from the test split."""
    import hashlib
    test = [r for r in iter_jsonl(manifest) if r["split"] == "test"]
    test.sort(key=lambda r: hashlib.sha256(("vlm_pilot:" + r["item_id"]).encode()).hexdigest())
    positives = [r for r in test if r["kind"] == "positive"][: size // 2]
    matched = [r for r in test if r["kind"] == "negative"
               and r["covariates"].get("matched_same_source")][: size // 4]
    others = [r for r in test if r["kind"] == "negative"
              and not r["covariates"].get("matched_same_source")][: size - len(positives) - len(matched)]
    return positives + matched + others


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--pilot-size", type=int, default=500)
    parser.add_argument("--frames", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-model-len", type=int, default=16384)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        done = {r["item_id"] for r in iter_jsonl(args.out)}
    items = [r for r in select_pilot(args.manifest, args.pilot_size)
             if r["item_id"] not in done]
    print(f"pilot items to score: {len(items)} (done: {len(done)})", flush=True)
    if not items:
        return 0

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model_path, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, limit_mm_per_prompt={"image": args.frames})
    sampling = SamplingParams(temperature=0.0, max_tokens=300)
    scored = failed = no_frames = 0
    with args.out.open("a") as out, tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            conversations, kept = [], []
            for item in batch:
                clip = Path(item["clip_action"])
                frames = extract_frames(args.ffmpeg, clip, args.frames, tmp_dir) if clip.is_file() else []
                if not frames:
                    no_frames += 1
                    out.write(json.dumps({"item_id": item["item_id"],
                                          "error": "no_frames",
                                          "runner_version": RUNNER_VERSION}) + "\n")
                    continue
                conversations.append([
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": frame_message(frames)},
                ])
                kept.append(item)
            if not conversations:
                continue
            outputs = llm.chat(conversations, sampling)
            for item, output in zip(kept, outputs):
                text = output.outputs[0].text if output.outputs else ""
                row = {"item_id": item["item_id"], "kind": item["kind"],
                       "label_reaction_present": item["label_reaction_present"],
                       "label_reaction_strength": item.get("label_reaction_strength"),
                       "label_responder_role": item.get("label_responder_role"),
                       "covariates": item["covariates"],
                       "runner_version": RUNNER_VERSION}
                try:
                    row["result"] = parse_result(text)
                    scored += 1
                except (ValueError, json.JSONDecodeError, TypeError) as error:
                    row["error"] = f"{type(error).__name__}: {error}"
                    row["raw_output"] = text[:500]
                    failed += 1
                out.write(json.dumps(row, sort_keys=True) + "\n")
            out.flush()
            print(f"progress {min(start + args.batch_size, len(items))}/{len(items)} "
                  f"(scored={scored} failed={failed} no_frames={no_frames})", flush=True)
    print(json.dumps({"runner_version": RUNNER_VERSION, "scored": scored,
                      "failed": failed, "no_frames": no_frames}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
