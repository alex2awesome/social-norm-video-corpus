#!/usr/bin/env python3
"""Freeze a balanced blind audit of witnessed authority-reaction text cues."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import cv2

if __package__:
    from scripts.render_full_corpus_score_audit import make_item_sheet, make_superpages
    from scripts.score_visual_scene_baselines import sample_frames
else:
    from render_full_corpus_score_audit import make_item_sheet, make_superpages
    from score_visual_scene_baselines import sample_frames


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def cue_stratum(row: dict[str, Any]) -> str:
    cue_ids = row.get("authority_cue_ids") or []
    if any(cue.startswith("phrase:") for cue in cue_ids):
        return "phrase"
    if any(cue.startswith("tag:") for cue in cue_ids):
        return "tag"
    return "norm"


def negative_stratum(row: dict[str, Any]) -> str:
    role = str(row.get("reactor_role_provenance") or "unknown")
    return role if role in {"camera_person", "mixed", "victim", "bystander"} else "other"


def balanced_source_disjoint(
    rows: list[dict[str, Any]],
    positives: int,
    negatives: int,
    seed: str,
    excluded_uids: set[str] | None = None,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    used_uids: set[str] = set(excluded_uids or ())

    def choose(candidates: list[dict[str, Any]], count: int, key) -> list[dict[str, Any]]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in candidates:
            buckets[key(row)].append(row)
        queues = []
        for name in sorted(buckets):
            rng.shuffle(buckets[name])
            queues.append((name, deque(buckets[name])))
        chosen = []
        while queues and len(chosen) < count:
            next_round = []
            for name, queue in queues:
                row = None
                while queue:
                    candidate = queue.popleft()
                    if candidate["uid"] not in used_uids:
                        row = candidate
                        break
                if row is not None:
                    chosen.append(row)
                    used_uids.add(row["uid"])
                if queue:
                    next_round.append((name, queue))
                if len(chosen) == count:
                    break
            queues = next_round
        if len(chosen) != count:
            raise ValueError(f"wanted {count} source-disjoint rows, found {len(chosen)}")
        return chosen

    positive_rows = [row for row in rows if row["authority_reaction_cue"]]
    # Hard controls emphasize camera-person/mixed roles, where authority or
    # participant speech is most easily confused with a bystander reaction.
    negative_rows = sorted(
        (row for row in rows if not row["authority_reaction_cue"]),
        key=lambda row: (
            row.get("reactor_role_provenance") not in {"camera_person", "mixed"},
            -len(row.get("reaction_observations") or []),
        ),
    )
    selected = choose(positive_rows, positives, cue_stratum)
    selected.extend(choose(negative_rows, negatives, negative_stratum))
    rng.shuffle(selected)
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--positives", type=int, default=24)
    parser.add_argument("--negatives", type=int, default=24)
    parser.add_argument("--seed", default="witnessed-authority-cue-audit-v1")
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument(
        "--exclude-sealed",
        type=Path,
        action="append",
        default=[],
        help="Exclude every UID present in a prior sealed JSONL selection.",
    )
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite frozen audit: {args.out}")
    if args.frames != 12:
        raise SystemExit("frozen sheet layout requires 12 frames")

    rows = load_jsonl(args.scores)
    excluded_uids = {
        str(row["uid"])
        for path in args.exclude_sealed
        for row in load_jsonl(path)
    }
    selected = balanced_source_disjoint(
        rows,
        args.positives,
        args.negatives,
        args.seed,
        excluded_uids=excluded_uids,
    )
    args.out.mkdir(parents=True)
    sheets_dir = args.out / "blind_item_sheets"
    sheets_dir.mkdir()
    root = args.project_root.resolve()
    sealed_rows = []
    blind_rows = []
    item_sheets = []
    for audit_index, row in enumerate(selected):
        source = root / "data" / "hits" / row["uid"] / row["clip_name"]
        error = None
        timestamps: list[float | None] = []
        try:
            frames, media = sample_frames(source, args.frames)
            timestamps = media["sampled_timestamps"]
            sheet = make_item_sheet(frames, timestamps, audit_index)
            sheet_path = sheets_dir / f"{audit_index:04d}.jpg"
            cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
            item_sheets.append((audit_index, sheet))
        except Exception as exc:  # Frozen failure record, never silent skip.
            error = f"{type(exc).__name__}: {exc}"
            sheet_path = None
        blind_rows.append(
            {
                "audit_index": audit_index,
                "item_id": f"authority_cue_audit:{audit_index:04d}",
                "frame_count": len(timestamps),
                "sample_timestamps": timestamps,
                "sheet_path": str(sheet_path) if sheet_path else None,
                "render_error": error,
            }
        )
        sealed_rows.append(
            {
                **row,
                "audit_index": audit_index,
                "source_clip": str(source),
                "source_exists": source.is_file(),
                "render_error": error,
            }
        )

    pages = make_superpages(item_sheets, args.out / "blind_review_pages")
    blind_path = args.out / "blind_manifest.jsonl"
    sealed_path = args.out / "sealed_selection.jsonl"
    blind_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in blind_rows)
    )
    sealed_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in sealed_rows)
    )
    manifest = {
        "schema_version": 1,
        "kind": "witnessed_authority_reaction_cue_blind_audit",
        "seed": args.seed,
        "items": len(selected),
        "positive_cue_items": sum(row["authority_reaction_cue"] for row in selected),
        "negative_control_items": sum(not row["authority_reaction_cue"] for row in selected),
        "unique_sources": len({row["uid"] for row in selected}),
        "excluded_prior_sources": len(excluded_uids),
        "blind_manifest_sha256": hashlib.sha256(blind_path.read_bytes()).hexdigest(),
        "sealed_selection_sha256": hashlib.sha256(sealed_path.read_bytes()).hexdigest(),
        "selection_sha256": sha256_json(
            [(row["item_id"], row["authority_reaction_cue"]) for row in selected]
        ),
        "review_order": "blind visual scene review before sealed cue/reaction reveal",
        "pages": pages,
        "corpus_mutated": False,
    }
    (args.out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
