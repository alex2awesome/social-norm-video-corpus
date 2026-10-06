#!/usr/bin/env python3
"""Build the R1 reaction-prediction task manifest (positives + controls).

Positives: witnessed action clips whose source-context label is organic
capture or genuine-reaction staged content in a social norm domain; targets
are reaction presence (1), strength, and responder role, with the reaction
itself excluded from the clip.  Controls: no-reaction negative clips,
preferring SAME-SOURCE matches (channel/style/scene held constant), then
unmatched fill to balance.  Splits are source-disjoint via the label-model
splitter.  Append-only manifest; no media is touched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from weaksup.label_model_v1 import source_disjoint_split

MANIFEST_VERSION = "reaction_prediction_manifest_v1"

POSITIVE_CONTENT_TYPES = {"organic_capture", "staged_prank_or_experiment"}
SOCIAL_DOMAINS = {"interpersonal_social", "traffic_driving", "property"}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_source_context(path: Path) -> dict[str, dict[str, Any]]:
    labels = {}
    for row in iter_jsonl(path):
        if row.get("result"):
            labels[row["uid"]] = row["result"]
    return labels


def positive_items(
    cut_manifests: list[Path],
    proposals: dict[str, dict[str, Any]],
    context: dict[str, dict[str, Any]],
    root: Path,
) -> list[dict[str, Any]]:
    items = []
    seen = set()
    for manifest in cut_manifests:
        for row in iter_jsonl(manifest):
            if not row.get("ok") or row.get("view") == "context":
                continue
            item_id = row["item_id"]
            if item_id in seen:
                continue
            seen.add(item_id)
            uid = row["uid"]
            label = context.get(uid)
            if not label or label["content_type"] not in POSITIVE_CONTENT_TYPES:
                continue
            if label["norm_domain"] not in SOCIAL_DOMAINS:
                continue
            try:
                metadata = json.loads(
                    (root / "data" / "hits" / uid / "metadata.json").read_text()
                )
            except (OSError, json.JSONDecodeError):
                continue
            scene = (metadata.get("provenance") or {}).get("scene") or {}
            items.append({
                "item_id": item_id, "uid": uid, "kind": "positive",
                "clip_action": row["dest"],
                "clip_context": str(Path(row["dest"]).parent.parent.parent /
                                    "clips_context" / uid /
                                    f"clip_{row['clip_idx']}_context.mp4"),
                "label_reaction_present": 1,
                "label_reaction_strength": scene.get("reaction_strength"),
                "label_responder_role": scene.get("reactor_role"),
                "covariates": {
                    "severity": scene.get("severity"),
                    "n_people": scene.get("n_people"),
                    "scene_type": scene.get("scene_type"),
                    "content_type": label["content_type"],
                    "norm_domain": label["norm_domain"],
                    "inferred_norm": label.get("inferred_norm"),
                    "platform": uid.split("__")[0],
                    "evidence_tier": proposals.get(item_id, {}).get("evidence_tier"),
                },
            })
    return items


def negative_items(
    root: Path, positive_uids: set[str], max_items: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    matched, unmatched = [], []
    for directory in sorted((root / "data" / "negatives").iterdir()):
        if not directory.is_dir():
            continue
        uid = directory.name
        clips = sorted(directory.glob("*.mp4"))
        if not clips:
            continue
        try:
            metadata = json.loads((directory / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            metadata = {}
        item = {
            "item_id": f"negative:{uid}:{clips[0].stem}",
            "uid": uid, "kind": "negative",
            "clip_action": str(clips[0]), "clip_context": None,
            "label_reaction_present": 0,
            "label_reaction_strength": None, "label_responder_role": None,
            "covariates": {
                "negative_tier": metadata.get("modality") or metadata.get("label"),
                "matched_same_source": uid in positive_uids,
                "platform": uid.split("__")[0],
            },
        }
        (matched if uid in positive_uids else unmatched).append(item)
    kept = matched + unmatched[: max(0, max_items - len(matched))]
    return kept, {"matched": len(matched), "unmatched_kept": len(kept) - len(matched)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--source-context", type=Path, required=True)
    parser.add_argument("--cut-manifest-glob", default="data/action_clips_v1/cut_manifest_shard_*.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    proposals = {r["item_id"]: r for r in iter_jsonl(args.proposals)}
    context = load_source_context(args.source_context)
    cut_manifests = sorted(args.root.glob(args.cut_manifest_glob))
    positives = positive_items(cut_manifests, proposals, context, args.root)
    negatives, negative_counts = negative_items(
        args.root, {p["uid"] for p in positives}, max_items=len(positives)
    )
    items = positives + negatives
    split = source_disjoint_split(
        [i["item_id"] for i in items],
        {i["item_id"]: i["uid"] for i in items},
        salt="reaction_prediction_v1",
    )
    with args.out.open("x") as out:
        for item in items:
            out.write(json.dumps({
                **item, "split": split[item["item_id"]],
                "manifest_version": MANIFEST_VERSION,
            }, sort_keys=True) + "\n")
    summary = {
        "manifest_version": MANIFEST_VERSION,
        "positives": len(positives), "negatives": len(negatives),
        **{f"negatives_{k}": v for k, v in negative_counts.items()},
        "splits": {name: sum(1 for i in items if split[i["item_id"]] == name)
                   for name in ("train", "calibration", "test")},
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
