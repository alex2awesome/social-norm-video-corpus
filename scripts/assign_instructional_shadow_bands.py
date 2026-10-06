#!/usr/bin/env python3
"""Join independent instructional shadow scores into recoverable review bands.

Bands are queues, never corpus dispositions. In particular, low-evidence bands
must not be deleted: model disagreement, animation, subtle dialogue, and broad
localization are known failure modes requiring sampled manual review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_successes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        row["item_id"]: row
        for row in rows
        if row.get("result") is not None and row.get("error") is None
    }


def yes(result: dict[str, Any] | None, field: str) -> bool:
    return bool(result and result.get(field) == "yes")


def scene_vote(row: dict[str, Any] | None) -> bool:
    if not row:
        return False
    result = row["result"]
    return yes(result, "usable_demo_after_relabel") and yes(
        result, "social_norm_domain"
    )


def assign_band(
    qwen: dict[str, Any] | None,
    glm: dict[str, Any] | None,
    text: dict[str, Any] | None,
    duration_hint: float = 0.0,
) -> str:
    """Assign one deterministic non-destructive queue."""
    if qwen is None or glm is None or text is None:
        return "incomplete_score_coverage"
    qr, gr, tr = qwen["result"], glm["result"], text["result"]
    q_scene, g_scene = scene_vote(qwen), scene_vote(glm)
    text_social = yes(tr, "social_norm_candidate")
    both_clean = (
        qr.get("localization_quality") == "clean"
        and gr.get("localization_quality") == "clean"
    )
    both_label = yes(qr, "proposed_norm_supported") and yes(
        gr, "proposed_norm_supported"
    )

    if q_scene and g_scene:
        if both_clean and both_label and text_social:
            if duration_hint > 180:
                return "dual_clean_long_context_review"
            return "dual_clean_label_candidate"
        if both_clean and text_social:
            return "dual_clean_relabel_review"
        if both_clean:
            return "dual_clean_text_conflict"
        return "dual_scene_recut_review"
    if q_scene or g_scene:
        return "single_vlm_candidate_review"
    if text_social:
        return "text_positive_visual_rescue_review"
    return "low_evidence_manual_sample"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--glm", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument(
        "--proxy-manifest",
        type=Path,
        help=(
            "Optional JSON proxy manifest whose max_frames/sampling_fps values "
            "supply duration when the source manifest predates duration_hint."
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen band file: {args.out}")

    manifest = load_jsonl(args.manifest)
    qwen = latest_successes(load_jsonl(args.qwen))
    glm = latest_successes(load_jsonl(args.glm))
    text = latest_successes(load_jsonl(args.text))
    proxy_durations: dict[str, float] = {}
    if args.proxy_manifest:
        payload = json.loads(args.proxy_manifest.read_text())
        for record in payload.get("records") or []:
            try:
                proxy_durations[record["item_id"]] = (
                    float(record["max_frames"]) / float(record["sampling_fps"])
                )
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
    counts: Counter[str] = Counter()
    records = []
    for row in manifest:
        item_id = row["item_id"]
        duration_hint = float(
            row.get("duration_hint") or proxy_durations.get(item_id) or 0
        )
        band = assign_band(
            qwen.get(item_id),
            glm.get(item_id),
            text.get(item_id),
            duration_hint,
        )
        counts[band] += 1
        records.append(
            {
                "ordinal": row["ordinal"],
                "item_id": item_id,
                "uid": row["uid"],
                "polarity": row.get("polarity"),
                "category": row.get("category"),
                "norm": row.get("norm"),
                "duration_hint": duration_hint,
                "band": band,
                "qwen_result": qwen.get(item_id, {}).get("result"),
                "glm_result": glm.get(item_id, {}).get("result"),
                "text_result": text.get(item_id, {}).get("result"),
                "corpus_disposition": None,
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in records)
    )
    summary_path = args.summary or args.out.with_suffix(".summary.json")
    payload = {
        "schema_version": 1,
        "kind": "instructional_shadow_score_bands",
        "records": len(records),
        "bands": dict(sorted(counts.items())),
        "coverage": {
            "qwen": len(qwen),
            "glm": len(glm),
            "text": len(text),
        },
        "source_manifest_sha256": sha256_file(args.manifest),
        "policy": "shadow_only_non_destructive",
        "manual_band_audit_required": True,
        "corpus_mutated": False,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
