"""Cut the pre-reaction window for each match with ffmpeg.

For a reaction at t_react we save [t_react - pre, t_react + post] (defaults
12s before, 2s after). We first try a fast stream copy seeked to the nearest
keyframe; if that yields an empty/!valid file we fall back to a precise
re-encode (`-ss` after `-i`, libx264 + aac).
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from . import state
from .media_integrity import probe_media_timing, timing_flags

log = logging.getLogger("clip")


def _duration_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1024:
        return False
    try:
        timing = probe_media_timing(path, timeout=30)
        return (
            timing["timeline_duration_sec"] > 0.1
            and not timing_flags(timing)
        )
    except (ValueError, subprocess.SubprocessError):
        return False


def _cut_copy(src: Path, dst: Path, ss: float, to: float) -> bool:
    dur = max(0.1, to - ss)
    cmd = ["ffmpeg", "-y", "-ss", f"{ss:.3f}", "-i", str(src),
           "-t", f"{dur:.3f}", "-c", "copy", "-avoid_negative_ts", "make_zero",
           str(dst)]
    subprocess.run(cmd, capture_output=True, timeout=300)
    return _duration_ok(dst)


def _cut_reencode(src: Path, dst: Path, ss: float, to: float) -> bool:
    dur = max(0.1, to - ss)
    cmd = ["ffmpeg", "-y", "-i", str(src), "-ss", f"{ss:.3f}",
           "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "veryfast",
           "-crf", "20", "-c:a", "aac", "-movflags", "+faststart", str(dst)]
    subprocess.run(cmd, capture_output=True, timeout=600)
    return _duration_ok(dst)


def _probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30)
        return float(out.stdout.strip() or 0)
    except (ValueError, subprocess.SubprocessError):
        return 0.0


def sample_negatives(video_path: Path, video_id: str, matches: list[dict],
                     cfg: dict, url: str = None, prov: dict = None,
                     only_no_violation: bool = False) -> list[dict]:
    """For each positive pre-reaction window, cut a same-length NEGATIVE clip from
    elsewhere in the SAME video that does NOT precede a reaction -- a matched
    control for the anticipation task (same camera/setting -> the model can't
    cheat on video-level features). Windows are non-overlapping and kept a guard
    distance from every reaction (so they aren't lead-up or aftermath). 1:1 with
    positives, capped, spread across the video. Returns the saved negatives."""
    ncfg = cfg["clip"].get("negatives", {})
    if not ncfg.get("enabled", False) or not matches:
        return []
    clip_cfg = cfg["clip"]
    pre = clip_cfg.get("pre_seconds", 20)
    post = clip_cfg.get("post_seconds", 2)
    L = pre + post
    guard = ncfg.get("guard_sec", 20)
    reencode = clip_cfg.get("reencode_fallback", True)

    dur = _probe_duration(video_path)
    if dur > ncfg.get("max_source_duration_sec", 1e9):
        return []  # likely a compilation -> "elsewhere in the video" isn't event-free
    if dur < 2 * L:
        return []
    reactions = sorted(float(m["start"]) for m in matches)
    # exclusion zone per reaction: the positive window [r-pre, r+post] + guard buffer
    zones = [(r - pre - guard, r + post + guard) for r in reactions]

    # non-overlapping candidate windows clear of every exclusion zone
    cands, a = [], 0.0
    while a + L <= dur:
        if all(a + L <= z0 or a >= z1 for z0, z1 in zones):
            cands.append(a)
        a += L
    if not cands:
        return []

    want = min(ncfg.get("max_per_video", 12),
               max(1, ncfg.get("per_positive", 1) * len(matches)), len(cands))
    if want >= len(cands):
        chosen = cands
    else:  # evenly spaced across the candidate slots (reproducible, well-spread)
        step = len(cands) / want
        chosen = [cands[int(i * step)] for i in range(want)]
    if only_no_violation:
        chosen = []   # backfill mode: skip re-cutting existing no_reaction clips

    neg_dir = state.resolve_path(cfg["paths"].get("negatives", "data/negatives")) / video_id
    neg_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for j, a in enumerate(chosen):
        dst = neg_dir / f"clip_{j}.mp4"
        ok = _cut_copy(video_path, dst, a, a + L)
        if not ok and reencode:
            ok = _cut_reencode(video_path, dst, a, a + L)
        if ok:
            saved.append({"neg_idx": j, "window": [round(a, 3), round(a + L, 3)]})
    if saved:
        (neg_dir / "metadata.json").write_text(json.dumps({
            "video_id": video_id, "url": url, "label": "no_reaction",
            # provenance mirrors the parent witnessed positive (same source video):
            "provenance": {**(prov or {}), "modality": "negative_control"},
            "n_negatives": len(saved), "duration": round(dur, 1),
            "negatives": saved,
            "positive_windows": [m.get("clip_window") for m in matches],
            "reaction_times": [round(r, 3) for r in reactions],
        }, indent=2, ensure_ascii=False))
    # ---- tier 1: within-video no_violation negatives (reaction-free AND
    #      language-clean windows). Approximate label: keyword-cleanliness is not
    #      a guarantee of no violation (silent/neutral-worded violations slip
    #      through) -- tier-2 null_verified is the detector-verified counterpart.
    nv_saved = []
    nvcfg = ncfg.get("no_violation", {}) or {}
    if nvcfg.get("enabled", False) and cands:
        tpath = state.resolve_path(cfg["paths"].get("transcripts", "data/transcripts")) / f"{video_id}.json"
        transcript = None
        if tpath.exists():
            try:
                transcript = json.load(open(tpath, encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                transcript = None
        if transcript and transcript.get("words"):
            from . import detect_reactions   # lazy: avoid any import cycle
            kw = detect_reactions.ReactionDetector(cfg)
            words = transcript["words"]
            min_win = float(nvcfg.get("min_window_sec", 6))
            clean = []
            for a in cands:
                if L < min_win:
                    continue
                window_words = [w for w in words
                                if w.get("start", 0) >= a and w.get("end", 0) <= a + L]
                # language-clean == the disapproval lexicon finds nothing in this window
                if not kw.detect({"words": window_words}):
                    clean.append(a)
            if clean:
                nv_want = min(int(nvcfg.get("max_per_video", 6)),
                              int(nvcfg.get("per_positive", 2)) * len(matches),
                              len(clean))
                if nv_want >= len(clean):
                    nv_chosen = clean
                else:
                    step = len(clean) / nv_want
                    nv_chosen = [clean[int(i * step)] for i in range(nv_want)]
                for k, a in enumerate(nv_chosen):
                    dst = neg_dir / f"nv_{k}.mp4"
                    ok = _cut_copy(video_path, dst, a, a + L)
                    if not ok and reencode:
                        ok = _cut_reencode(video_path, dst, a, a + L)
                    if ok:
                        nv_saved.append({"neg_idx": k, "window": [round(a, 3), round(a + L, 3)]})
    if nv_saved:
        mpath = neg_dir / "metadata.json"
        meta = json.load(open(mpath)) if mpath.exists() else {
            "video_id": video_id, "url": url, "label": "no_violation",
            "provenance": {**(prov or {}), "modality": "negative_control_nv"},
            "duration": round(dur, 1)}
        meta["no_violation_negatives"] = nv_saved
        mpath.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        log.info("sampled %d no_violation negatives for %s", len(nv_saved), video_id)

    log.info("sampled %d negatives for %s (%d positives, dur %.0fs)",
             len(saved), video_id, len(matches), dur)
    return saved


def save_null_verified(video_path: Path, video_id: str, cfg: dict,
                       url: str = None, category: str = None,
                       prov: dict = None) -> list[dict]:
    """Cut matched-length NULL clips from a detector-cleared non-violating video
    (curated null queries; see batch_detect._route). Mirrors positive clip
    geometry (pre/post seconds) so negatives are temporally comparable to
    positives. Windows are evenly spaced across the video. Writes
    data/negatives/{uid}/null_{k}.mp4 + metadata.json (label no_violation,
    modality null_verified). Returns the saved windows."""
    clip_cfg = cfg["clip"]
    pre = clip_cfg.get("pre_seconds", 20)
    post = clip_cfg.get("post_seconds", 2)
    L = pre + post
    ncfg = (cfg["clip"].get("negatives", {}) or {}).get("no_violation", {}) or {}
    reencode = clip_cfg.get("reencode_fallback", True)
    dur = _probe_duration(video_path)
    if dur < L:
        return []
    max_clips = int(ncfg.get("max_per_video", 6))
    n_fit = max(1, int(dur // L))
    want = max(0, min(max_clips, n_fit))
    if want == 0:
        return []
    neg_dir = state.resolve_path(cfg["paths"].get("negatives", "data/negatives")) / video_id
    neg_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for k in range(want):
        center = dur * (k + 1) / (want + 1)
        a = max(0.0, min(center - L / 2, dur - L))
        dst = neg_dir / f"null_{k}.mp4"
        ok = _cut_copy(video_path, dst, a, a + L)
        if not ok and reencode:
            ok = _cut_reencode(video_path, dst, a, a + L)
        if ok:
            saved.append({"neg_idx": k, "window": [round(a, 3), round(a + L, 3)]})
    if saved:
        # MERGE with any existing metadata (defensive: a uid could in principle
        # already carry within-video negatives). Dedicated key keeps the three
        # negative tiers cleanly separated: negatives (no_reaction) /
        # no_violation_negatives (tier 1) / null_negatives (tier 2, verified).
        mpath = neg_dir / "metadata.json"
        meta = json.load(open(mpath)) if mpath.exists() else {}
        meta.update({
            "video_id": video_id, "url": url, "label": "no_violation",
            "modality": "null_verified", "category": category,
            "provenance": {**(prov or {}), "modality": "null_verified",
                           "verified_by": "candid_detector"},
            "duration": round(dur, 1),
        })
        meta["null_negatives"] = saved
        meta["n_null"] = len(saved)
        mpath.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        log.info("saved %d null_verified clips for %s (dur %.0fs)",
                 len(saved), video_id, dur)
    return saved


def retain_discussion_video(video_path: Path, video_id: str, cfg: dict) -> Path:
    """Move a commentary source video out of purgeable raw storage."""
    if not video_path.exists() or video_path.stat().st_size <= 0:
        raise FileNotFoundError(f"missing commentary source video: {video_path}")
    out_dir = state.resolve_path(
        cfg["paths"].get("discussion_video", "data/discussion_video")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = video_path.suffix.lower() or ".mp4"
    out = out_dir / f"{video_id}{suffix}"
    if out.exists() and out.stat().st_size > 0:
        return out
    try:
        video_path.replace(out)
    except OSError:
        tmp = out.with_suffix(out.suffix + ".tmp")
        shutil.copy2(video_path, tmp)
        tmp.replace(out)
        video_path.unlink()
    return out


def save_discussion(video_id: str, matches: list[dict], cfg: dict,
                    cand: dict, category: str = None, agent: str = None,
                    prov: dict = None, source_video: Path = None) -> Path:
    """Save commentary weak labels and the stable source-video reference."""
    disc_dir = state.resolve_path(cfg["paths"].get("discussion", "data/discussion"))
    disc_dir.mkdir(parents=True, exist_ok=True)
    rec = {
        "video_id": video_id,
        "url": cand.get("url"),
        "title": cand.get("title"),
        "source": cand.get("source"),
        "category": category,
        "agent": agent or "human",   # human | non_human (autonomous vehicle / robot)
        "modality": "commentary",
        "source_video": source_video.name if source_video is not None else None,
        "provenance": {**(prov or {}), "modality": "commentary", "agent": agent or "human"},
        "n_statements": len(matches),
        "statements": [
            {"quote": m.get("matched_text") or m.get("phrase"),
             "norm": m.get("norm"), "signal": m.get("tag"),
             "start": m.get("start"), "end": m.get("end")}
            for m in matches
        ],
    }
    out = disc_dir / f"{video_id}.json"
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    return out


def save_instructional(video_path: Path, video_id: str, result: dict, cfg: dict,
                       cand: dict, category: str = None, prov: dict = None) -> int:
    """Save a DIDACTIC norm video to the instructional corpus: structured extraction
    (norms + demo spans + the educator's explanation = ground-truth label) plus a
    cut clip per locatable demo span. Returns the number of demo clips saved."""
    out_dir = state.resolve_path(cfg["paths"].get("instructional", "data/instructional")) / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    reencode = cfg["clip"].get("reencode_fallback", True)
    pad = 1.0
    n_clips = 0
    for idx, d in enumerate(result.get("demos", [])):
        ss, to = d.get("start"), d.get("end")
        if ss is None or to is None or to <= ss:
            continue
        dst = out_dir / f"demo_{idx}.mp4"
        ok = _cut_copy(video_path, dst, max(0.0, ss - pad), to + pad)
        if not ok and reencode:
            ok = _cut_reencode(video_path, dst, max(0.0, ss - pad), to + pad)
        if ok:
            d["clip"] = dst.name
            n_clips += 1
    rec = {
        "video_id": video_id, "url": cand.get("url"), "title": cand.get("title"),
        "source": cand.get("source"), "category": category,
        "modality": "instructional", "genre": result.get("genre"),
        "provenance": {**(prov or {}), "modality": "instructional"},
        "norms": result.get("norms"), "demos": result.get("demos"),
        "expected_behavior": result.get("expected_behavior"),
        "n_demo_clips": n_clips,
    }
    if cand.get("meta"):
        rec["source_meta"] = cand["meta"]
    (out_dir / "metadata.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False))
    log.info("INSTRUCTIONAL %s: %d norms, %d demos (%d clips), genre=%s",
             video_id, len(result.get("norms", [])), len(result.get("demos", [])),
             n_clips, result.get("genre"))
    return n_clips


def extract_clips(video_path: Path, video_id: str, matches: list[dict],
                  cfg: dict, url: str = None, agent: str = None,
                  prov: dict = None) -> list[dict]:
    """Cut one clip per match; write reaction_{idx}.txt + metadata.json.

    Mutates each match dict with `clip_idx`. Returns the kept matches.
    `video_id` is the uid; `url` is the canonical source page URL.
    """
    if not matches:
        return []
    clip_cfg = cfg["clip"]
    pre = clip_cfg.get("pre_seconds", 12)
    post = clip_cfg.get("post_seconds", 2)
    merge_win = clip_cfg.get("merge_window_sec", 22)   # reactions closer than this share one clip
    min_pre = clip_cfg.get("min_pre_roll_sec", 10)     # drop clips whose lead-up is truncated
    reencode = clip_cfg.get("reencode_fallback", True)

    hit_dir = state.resolve_path(cfg["paths"]["hits"]) / video_id
    hit_dir.mkdir(parents=True, exist_ok=True)

    # GROUP nearby reactions into one clip: 48% of consecutive reactions were <22s
    # apart, so per-reaction windows overlapped heavily (same moment saved twice).
    # One clip spans [first - pre, last + post] and carries every reaction in it.
    ms = sorted(matches, key=lambda m: m["start"])
    groups, cur = [], [ms[0]]
    for m in ms[1:]:
        if m["start"] - cur[-1]["start"] <= merge_win:
            cur.append(m)
        else:
            groups.append(cur)
            cur = [m]
    groups.append(cur)

    saved = []
    idx = 0
    for grp in groups:
        t0, t1 = grp[0]["start"], grp[-1]["start"]
        if t0 < min_pre:
            # reaction at the very head of the video -> the pre-violation lead-up
            # is missing; a truncated clip is useless for the anticipation task
            log.info("drop truncated group for %s (first reaction at %.1fs < %ds pre-roll)",
                     video_id, t0, min_pre)
            continue
        ss = max(0.0, t0 - pre)
        to = t1 + post
        dst = hit_dir / f"clip_{idx}.mp4"

        ok = _cut_copy(video_path, dst, ss, to)
        if not ok and reencode:
            log.info("copy cut failed for %s clip %d; re-encoding", video_id, idx)
            ok = _cut_reencode(video_path, dst, ss, to)
        if not ok:
            log.warning("could not cut clip %d for %s", idx, video_id)
            continue

        lines = []
        for m in grp:
            m["clip_idx"] = idx
            m["clip_window"] = [round(ss, 3), round(to, 3)]
            lines.append(
                "phrase:    {phrase}\n"
                "matched:   {matched_text}\n"
                "tier:      {tier}\n"
                "tag:       {tag}\n"
                "reaction:  {start:.3f}-{end:.3f}s\n"
                "clip:      {ws:.3f}-{we:.3f}s\n"
                "speaker:   {speaker}\n"
                "context:   {context}\n".format(
                    phrase=m["phrase"], matched_text=m.get("matched_text", ""),
                    tier=m["tier"], tag=m.get("tag"), start=m["start"], end=m["end"],
                    ws=ss, we=to, speaker=m.get("speaker"), context=m.get("context", "")))
            saved.append(m)
        (hit_dir / f"reaction_{idx}.txt").write_text("\n".join(lines))
        idx += 1

    if not saved:
        # every group dropped (truncated pre-roll / failed cuts) -> not a hit;
        # leave no empty dir behind (caller treats this as a miss)
        import shutil
        shutil.rmtree(hit_dir, ignore_errors=True)
        log.info("no clips kept for %s (%d reactions all dropped)", video_id, len(matches))
        return []

    meta = {
        "video_id": video_id,
        "url": url,
        "agent": agent or "human",   # human | non_human (autonomous vehicle / robot)
        # full provenance so each clip is self-describing (don't lump corpora/sources):
        "provenance": {**(prov or {}), "modality": "witnessed", "agent": agent or "human"},
        "n_clips": idx,                # clips on disk (a clip may hold several reactions)
        "n_reactions": len(saved),     # reactions kept across all clips
        "reactions": [
            {k: v for k, v in m.items() if k != "matched_text" or True}
            for m in saved
        ],
    }
    (hit_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False))
    log.info("saved %d clips (%d/%d reactions) for %s",
             idx, len(saved), len(matches), video_id)
    return saved
