"""Sample N% of witnessed hits for MANUAL quality review (the 5% QA loop).

Emits one compact record per sampled hit -- everything a reviewer needs to
judge quality WITHOUT opening every video: title, scene scores, agent/modality,
clip/reaction counts, audio-event peaks, each reaction's phrase+context, and a
transcript excerpt around the first reaction. Hits already recorded in the QA
journal (data/qa_journal.jsonl) are skipped, so repeated runs keep advancing
the 5% sample across NEW collection instead of re-drawing the same hits.

Deterministic: a hit is in the sample iff sha1(seed:uid) falls in the bottom
`rate` fraction -- stable across runs, no global RNG.

    ./run.sh python -u scripts/spotcheck_sample.py --rate 0.05 [--seed quality0614]
                                                   [--limit N] [--out FILE]
"""
import argparse
import hashlib
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, ".")
from src import state                                    # noqa: E402


def _in_sample(uid: str, rate: float, seed: str) -> bool:
    h = hashlib.sha1(f"{seed}:{uid}".encode()).digest()
    # first 4 bytes -> [0,1)
    frac = int.from_bytes(h[:4], "big") / 2**32
    return frac < rate


def _excerpt(transcript: dict, t: float, pad: float = 12.0) -> str:
    if not transcript or t is None:
        return ""
    segs = transcript.get("segments") or []
    out = [s.get("text", "").strip() for s in segs
           if s.get("end", 0) >= t - pad and s.get("start", 0) <= t + pad]
    return " ".join(x for x in out if x).strip()[:500]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.05)
    ap.add_argument("--seed", default="quality0614")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/qa_sample.jsonl")
    args = ap.parse_args()

    cfg = state.load_config()
    hits_dir = state.resolve_path(cfg["paths"]["hits"])
    ts_dir = state.resolve_path(cfg["paths"]["transcripts"])
    ae_dir = state.resolve_path(cfg.get("audio_events", {}).get("out_dir",
                                                                "data/audio_events"))
    journal_p = state.resolve_path("data/qa_journal.jsonl")
    reviewed = set()
    if journal_p.exists():
        for line in journal_p.read_text().splitlines():
            try:
                reviewed.add(json.loads(line)["uid"])
            except (json.JSONDecodeError, KeyError):
                pass

    conn = sqlite3.connect(state.resolve_path(cfg["paths"]["state_db"]), timeout=60)
    conn.execute("PRAGMA busy_timeout=60000")
    titles = dict(conn.execute(
        "SELECT video_id, title FROM seen_videos WHERE title IS NOT NULL"))
    conn.close()

    all_uids = sorted(p.name for p in hits_dir.iterdir() if p.is_dir())
    sample = [u for u in all_uids
              if u not in reviewed and _in_sample(u, args.rate, args.seed)]
    if args.limit:
        sample = sample[:args.limit]

    out_p = state.resolve_path(args.out)
    n = 0
    with out_p.open("w") as f:
        for uid in sample:
            mp = hits_dir / uid / "metadata.json"
            if not mp.exists():
                continue
            try:
                meta = json.loads(mp.read_text())
            except json.JSONDecodeError:
                continue
            prov = meta.get("provenance") or {}
            scene = prov.get("scene") or {}
            reactions = meta.get("reactions") or []
            first_t = min((r.get("start") for r in reactions
                           if r.get("start") is not None), default=None)
            transcript = {}
            tp = ts_dir / f"{uid}.json"
            if tp.exists():
                try:
                    transcript = json.loads(tp.read_text())
                except json.JSONDecodeError:
                    pass
            ae = {}
            aep = ae_dir / f"{uid}.json"
            if aep.exists():
                try:
                    ae = json.loads(aep.read_text()).get("summary") or {}
                except json.JSONDecodeError:
                    pass
            rec = {
                "uid": uid,
                "title": titles.get(uid, ""),
                "source": prov.get("platform"),
                "category": prov.get("category"),
                "query": prov.get("found_by_query"),
                "query_source": prov.get("query_source"),
                "agent": meta.get("agent"),
                "modality": prov.get("modality"),
                "severity": scene.get("severity"),
                "reaction_strength": scene.get("reaction_strength"),
                "scene_type": scene.get("scene_type"),
                "n_people": scene.get("n_people"),
                "violator_role": scene.get("violator_role"),
                "reactor_role": scene.get("reactor_role"),
                "n_clips": meta.get("n_clips"),
                "n_reactions": meta.get("n_reactions"),
                "audio_recovered": meta.get("audio_recovered", 0),
                "audio_interest": ae.get("interest"),
                "audio_scream": ae.get("scream_peak"),
                "audio_commotion": ae.get("commotion_peak"),
                "reactions": [
                    {"tag": r.get("tag"), "phrase": r.get("phrase"),
                     "norm": r.get("norm"), "start": r.get("start"),
                     "audio_peak": r.get("audio_peak"),
                     "context": (r.get("context") or "")[:160]}
                    for r in reactions[:8]
                ],
                "transcript_at_first_reaction": _excerpt(transcript, first_t),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1

    pool = len([u for u in all_uids if u not in reviewed])
    print(f"hits on disk: {len(all_uids)}  already reviewed: {len(reviewed)}  "
          f"unreviewed pool: {pool}")
    print(f"sampled {n} hits (rate {args.rate}, seed {args.seed}) -> {out_p}")


if __name__ == "__main__":
    main()
