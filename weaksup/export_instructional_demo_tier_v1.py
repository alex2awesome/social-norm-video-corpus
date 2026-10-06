#!/usr/bin/env python3
"""Export the audited instructional demo tier as a dataset-view manifest.

Selects demo items where the semantic (non-explanation polarity) and visual
(scene depiction) evidence families BOTH voted positive and eligibility gates
passed — the ~0.80-posterior operating point set by the 2026-08-22 18-item
frame audit (4/6 demo precision vs 0/6 in the semantics-only stratum).  Emits
an append-only manifest joining each demo's existing clip file, metadata, and
posterior; no media is cut, moved, or relabeled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

EXPORT_VERSION = "instructional_demo_tier_v1"
OPERATING_POINT_BASIS = (
    "20260822 frame audit: both-positive stratum 4/6 demos "
    "(2 exact labels), semantics-only 0/6; review-first tier, not acceptance"
)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def tier_items(posteriors_path: Path) -> list[dict[str, Any]]:
    selected = []
    for row in iter_jsonl(posteriors_path):
        votes = row["family_votes"]
        if (
            votes.get("explicit_semantics") == 1
            and votes.get("visual_depiction") == 1
            and not row["gate"]["failed_gates"]
            and not row["gate"]["unknown_gates"]
        ):
            selected.append(row)
    return selected


def export(root: Path, posteriors_path: Path, out_dir: Path,
           *, hash_media: bool = False) -> dict[str, Any]:
    out_path = out_dir / "instructional_demo_tier_manifest.jsonl"
    summary_path = out_dir / "summary.json"
    for path in (out_path, summary_path):
        if path.exists():
            raise FileExistsError(f"output exists: {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = tier_items(posteriors_path)
    written = missing_clip = 0
    videos: set[str] = set()
    with out_path.open("x") as handle:
        for row in rows:
            _, uid, demo = row["item_id"].split(":")
            idx = int(demo.rsplit("_", 1)[-1])
            clip = root / "data" / "instructional" / uid / f"demo_{idx}.mp4"
            if not clip.is_file() or clip.stat().st_size == 0:
                missing_clip += 1
                continue
            metadata = json.loads(
                (root / "data" / "instructional" / uid / "metadata.json").read_text()
            )
            demos = metadata.get("demos") or []
            demo_record = demos[idx] if idx < len(demos) else {}
            handle.write(json.dumps({
                "item_id": row["item_id"], "uid": uid, "demo_idx": idx,
                "clip_path": str(clip),
                "clip_sha256": _sha256(clip) if hash_media else None,
                "posterior_demonstration_present": row["posterior_positive"],
                "norm": demo_record.get("norm"),
                "polarity": demo_record.get("polarity"),
                "title": metadata.get("title"),
                "export_version": EXPORT_VERSION,
                "operating_point_basis": OPERATING_POINT_BASIS,
                "acceptance_label": None,
                "disposition": "review_first_candidate",
            }, sort_keys=True) + "\n")
            written += 1
            videos.add(uid)
    summary = {
        "export_version": EXPORT_VERSION,
        "tier_items": len(rows), "exported": written,
        "videos": len(videos), "missing_clip": missing_clip,
        "manifest_sha256": _sha256(out_path),
        "policy": "dataset_view_manifest_only_no_media_mutation",
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--posteriors", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--hash-media", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        export(args.root, args.posteriors, args.out_dir, hash_media=args.hash_media),
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
