#!/usr/bin/env python3
"""Export child clips from allowed and blocked historical snowball parents."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

if __package__:
    from src.snowball_scope import audit_snowball_parent
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.snowball_scope import audit_snowball_parent


def stable_key(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def parent_uid(native_id: str) -> str:
    return f"dailymotion__{native_id}"


def load_parent_metadata(repo: Path, uid: str) -> tuple[dict, dict]:
    path = repo / "data/hits" / uid / "metadata.json"
    if not path.exists():
        return {}, {}
    try:
        meta = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}, {}
    scene = (meta.get("provenance") or {}).get("scene") or meta.get("scene") or {}
    return meta, scene


def duration(path: Path, ffprobe: str) -> float:
    return float(subprocess.check_output(
        [ffprobe, "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)], text=True).strip())


def contact_sheet(media: Path, out: Path, ffmpeg: str, ffprobe: str) -> None:
    dur = duration(media, ffprobe)
    # Twelve uniformly spaced frames make role/action review substantially more
    # reliable than a midpoint while keeping the artifact compact.
    subprocess.run(
        [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", str(media),
         "-vf", f"fps=12/{dur},scale=240:-2,tile=4x3", "-frames:v", "1",
         "-y", str(out)], check=True)


def collect(repo: Path, per_band: int) -> tuple[list[dict], dict]:
    conn = sqlite3.connect(repo / "data/state.db")
    conn.row_factory = sqlite3.Row
    candidates = {"allow": [], "block": []}
    parent_counts = Counter()
    child_counts = Counter()
    queries = conn.execute(
        "SELECT query,category FROM queries WHERE platform='dmrelated' AND source='related'"
    ).fetchall()
    # Materialize the two joins once.  The live corpus has tens of thousands of
    # related queries; issuing one unindexed child scan per parent is both slow
    # and needlessly hard on the shared server.
    parents = {
        row["video_id"]: row for row in conn.execute(
            """SELECT video_id,title,n_reactions FROM seen_videos
               WHERE platform IN ('dailymotion','dmrelated')
                 AND modality='witnessed' AND is_hit=1"""
        ).fetchall()
    }
    children_by_query: dict[str, list] = {}
    for child in conn.execute(
        """SELECT video_id,title,query,category,query_source,modality,n_reactions
           FROM seen_videos WHERE platform='dmrelated' AND modality='witnessed'"""
    ).fetchall():
        children_by_query.setdefault(child["query"], []).append(child)
    for query_row in queries:
        uid = parent_uid(query_row["query"])
        parent = parents.get(uid)
        if parent is None:
            parent_counts["missing_parent"] += 1
            continue
        meta, scene = load_parent_metadata(repo, uid)
        decision = audit_snowball_parent(
            title=parent["title"] or "", agent=meta.get("agent"), scene=scene,
            reaction_count=int(parent["n_reactions"] or 0))
        band = "allow" if decision.allowed else "block"
        parent_counts[f"{band}:{decision.reason}"] += 1
        children = children_by_query.get(query_row["query"], [])
        child_counts[f"{band}:witnessed_children"] += len(children)
        for child in children:
            media = repo / "data/hits" / child["video_id"] / "clip_0.mp4"
            if not media.exists():
                continue
            candidates[band].append({
                "band": band, "parent_uid": uid, "parent_title": parent["title"],
                "parent_decision": decision.as_dict(), "child_uid": child["video_id"],
                "child_title": child["title"], "child_category": child["category"],
                "child_query_source": child["query_source"],
                "child_reaction_count": child["n_reactions"], "media": str(media),
            })
    selected = []
    for band in ("allow", "block"):
        ordered = sorted(candidates[band], key=lambda x: stable_key(x["child_uid"]))
        # Source-disjoint by child and parent; at most two children per parent so
        # one recommendation neighborhood cannot dominate the transfer audit.
        per_parent = Counter()
        seen_child = set()
        for item in ordered:
            if item["child_uid"] in seen_child or per_parent[item["parent_uid"]] >= 2:
                continue
            selected.append(item)
            seen_child.add(item["child_uid"])
            per_parent[item["parent_uid"]] += 1
            if sum(x["band"] == band for x in selected) >= per_band:
                break
    summary = {"historical_related_queries": len(queries),
               "parent_counts": dict(parent_counts), "child_counts": dict(child_counts),
               "selected": dict(Counter(x["band"] for x in selected)),
               "automatic_acceptance": False, "corpus_mutation_authorized": False}
    return selected, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-band", type=int, default=20)
    ap.add_argument("--ffmpeg", required=True)
    ap.add_argument("--ffprobe", required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    selected, summary = collect(args.repo, args.per_band)
    manifest = []
    for index, item in enumerate(selected):
        name = f"{item['band']}_{index:02d}_{item['child_uid']}.jpg"
        try:
            contact_sheet(Path(item["media"]), args.out / name, args.ffmpeg, args.ffprobe)
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
        manifest.append({**item, "contact_sheet": name})
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    (args.out / "population_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps({"rendered": len(manifest), **summary["selected"]}))


if __name__ == "__main__":
    main()
