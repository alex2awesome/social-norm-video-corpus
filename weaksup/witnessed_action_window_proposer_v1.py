#!/usr/bin/env python3
"""Propose exact pre-reaction action windows for witnessed clips.

For every joint-high witnessed clip (norm event supported AND reaction
grounded), propose the interval that should contain the violation itself:

    t_anchor     = earliest detected reaction start in the clip
    action_end   = t_anchor - guard (reaction excluded -> no label leakage)
    action_start = max(clip_start, t_anchor - pre_sec), snapped forward to
                   (last PANNs impact peak before t_anchor) - snap_lead when
                   an impact event (slap/crash/skid/...) exists in the window

Proposals are append-only shadow artifacts.  Nothing is cut: exact-artifact
review of a stratified sample must pass before any splice, per the standing
audit policy.  Known failure modes the sample must check: edits that reorder
action and reaction, and reactions to off-screen events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

PROPOSER_VERSION = "witnessed_action_window_proposer_v1"

# Impact-like commotion classes: sounds of the violent/chaotic event itself.
# Ambient commotion (Crowd, Hubbub, Siren, Car alarm) anchors nothing.
IMPACT_CLASSES = {
    "Slap, smack", "Whack, thwack", "Smash, crash", "Breaking", "Shatter",
    "Glass", "Gunshot, gunfire", "Skidding", "Tire squeal", "Squeal",
}

DEFAULTS = {
    "pre_sec": 10.0,
    "guard_sec": 0.3,
    "snap_lead_sec": 2.0,
    "min_duration_sec": 2.0,
    "impact_peak_min": 0.3,
}


def propose_action_window(
    *,
    item_id: str,
    reactions: list[dict[str, Any]],
    clip_window: tuple[float, float],
    audio_events: list[dict[str, Any]] | None = None,
    pre_sec: float = DEFAULTS["pre_sec"],
    guard_sec: float = DEFAULTS["guard_sec"],
    snap_lead_sec: float = DEFAULTS["snap_lead_sec"],
    min_duration_sec: float = DEFAULTS["min_duration_sec"],
    impact_peak_min: float = DEFAULTS["impact_peak_min"],
) -> dict[str, Any]:
    """Deterministic action-window proposal for one clip (source seconds)."""
    clip_start, clip_end = float(clip_window[0]), float(clip_window[1])
    starts = [
        float(reaction["start"])
        for reaction in reactions
        if isinstance(reaction.get("start"), (int, float))
    ]
    base = {
        "item_id": item_id,
        "proposer_version": PROPOSER_VERSION,
        "clip_window_sec": [clip_start, clip_end],
        "parameters": {
            "pre_sec": pre_sec, "guard_sec": guard_sec,
            "snap_lead_sec": snap_lead_sec,
            "min_duration_sec": min_duration_sec,
            "impact_peak_min": impact_peak_min,
        },
        "acceptance_label": None,
        "media_cut": False,
        "policy": "proposal_only_exact_artifact_review_required_before_splice",
    }
    if not starts:
        return {**base, "status": "no_reaction_timestamp", "action_window_sec": None}

    t_anchor = min(starts)
    action_end = t_anchor - guard_sec
    base_start = max(clip_start, t_anchor - pre_sec)

    snap = None
    for event in sorted(audio_events or [], key=lambda e: float(e.get("t0", 0))):
        if event.get("class") not in IMPACT_CLASSES:
            continue
        if float(event.get("peak") or 0) < impact_peak_min:
            continue
        t0 = float(event.get("t0", 0))
        if base_start <= t0 < t_anchor:
            snap = event  # keep the LAST qualifying impact before the anchor
    action_start = base_start
    snap_used = False
    if snap is not None:
        snapped = max(clip_start, float(snap["t0"]) - snap_lead_sec)
        # Snapping narrows the window toward the impact; never below minimum.
        if snapped > base_start and action_end - snapped >= min_duration_sec:
            action_start = snapped
            snap_used = True

    duration = action_end - action_start
    status = "proposed" if duration >= min_duration_sec else "short_window_review"
    return {
        **base,
        "status": status,
        "reaction_anchor_sec": t_anchor,
        "n_reaction_timestamps": len(starts),
        "action_window_sec": [round(action_start, 3), round(action_end, 3)],
        "action_window_clip_relative_sec": [
            round(action_start - clip_start, 3),
            round(action_end - clip_start, 3),
        ],
        "duration_sec": round(duration, 3),
        "audio_snap": {
            "used": snap_used,
            "class": snap.get("class") if snap else None,
            "peak": snap.get("peak") if snap else None,
            "impact_t0_sec": float(snap["t0"]) if snap else None,
        },
    }


def _high_band(path: Path) -> dict[str, float]:
    result = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row["shadow_band"] == "high_confidence_candidate":
                result[row["item_id"]] = row["posterior_positive"]
    return result


def select_items(
    model_dir: Path, selection: str = "joint_high"
) -> dict[str, dict[str, Any]]:
    """Witnessed items by evidence tier.

    ``joint_high``: high on BOTH norm event and reaction (strictest).
    ``any_high``: high on EITHER — the full-scale set; each item is tagged
    with which bands it holds so downstream tiers stay distinguishable.
    """
    norm = _high_band(model_dir / "posteriors_witnessed_norm_event_supported.jsonl")
    reaction = _high_band(model_dir / "posteriors_witnessed_reaction_grounded.jsonl")
    if selection == "joint_high":
        chosen = norm.keys() & reaction.keys()
    elif selection == "any_high":
        chosen = norm.keys() | reaction.keys()
    else:
        raise ValueError(f"invalid selection: {selection!r}")
    return {
        item: {
            "norm_event": norm.get(item),
            "reaction": reaction.get(item),
            "evidence_tier": "joint_high"
            if item in norm and item in reaction
            else "norm_event_only" if item in norm else "reaction_only",
        }
        for item in chosen
    }


def joint_high_items(model_dir: Path) -> dict[str, dict[str, Any]]:
    """Back-compatible strict selection."""
    return select_items(model_dir, "joint_high")


def clip_reactions(metadata: dict[str, Any], clip_idx: int) -> tuple[list, tuple | None]:
    reactions = [
        r for r in metadata.get("reactions") or []
        if str(r.get("clip_idx")) == str(clip_idx)
    ]
    windows = [
        r["clip_window"] for r in reactions
        if isinstance(r.get("clip_window"), list) and len(r["clip_window"]) == 2
    ]
    if not windows:
        return reactions, None
    return reactions, (
        min(float(w[0]) for w in windows), max(float(w[1]) for w in windows)
    )


def review_sample(
    proposals: list[dict[str, Any]], size: int, seed: str
) -> list[dict[str, Any]]:
    """Deterministic stratified sample: snap-used / no-snap / short-window."""
    strata: dict[str, list[dict[str, Any]]] = {"snapped": [], "unsnapped": [], "short": []}
    for row in proposals:
        if row["status"] == "short_window_review":
            strata["short"].append(row)
        elif row.get("audio_snap", {}).get("used"):
            strata["snapped"].append(row)
        elif row["status"] == "proposed":
            strata["unsnapped"].append(row)
    per = max(1, size // 3)
    sample = []
    for name, rows in strata.items():
        rows.sort(key=lambda r: hashlib.sha256(f"{seed}:{r['item_id']}".encode()).hexdigest())
        sample.extend(
            {
                **row,
                "review_stratum": name,
                "action_visible_in_window": "",
                "action_matches_label": "",
                "reaction_excluded": "",
                "edit_reorders_action_and_reaction": "",
                "boundary_adjustment_sec": "",
                "reviewer": "",
                "review_complete": False,
            }
            for row in rows[:per]
        )
    return sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True,
                        help="v5 model dir with witnessed posteriors")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--review-sample", type=int, default=24)
    parser.add_argument("--seed", default="action_window_v1_wave1")
    parser.add_argument("--selection", choices=("joint_high", "any_high"),
                        default="joint_high")
    args = parser.parse_args()
    out_proposals = args.out_dir / "action_window_proposals.jsonl"
    out_sample = args.out_dir / "review_sample.jsonl"
    out_summary = args.out_dir / "summary.json"
    for path in (out_proposals, out_sample, out_summary):
        if path.exists():
            raise FileExistsError(f"output exists: {path}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    items = select_items(args.model_dir, args.selection)
    proposals = []
    counts = {"items": len(items), "selection": args.selection,
              "missing_metadata": 0, "no_clip_window": 0}
    for item_id, posteriors in sorted(items.items()):
        _, uid, clip_name = item_id.split(":")
        clip_idx = clip_name.rsplit("_", 1)[-1]
        metadata_path = args.root / "data" / "hits" / uid / "metadata.json"
        if not metadata_path.is_file():
            counts["missing_metadata"] += 1
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        reactions, window = clip_reactions(metadata, int(clip_idx))
        if window is None:
            counts["no_clip_window"] += 1
            continue
        audio_path = args.root / "data" / "audio_events" / f"{uid}.json"
        events = []
        if audio_path.is_file():
            try:
                events = json.loads(audio_path.read_text()).get("events") or []
            except (json.JSONDecodeError, UnicodeDecodeError):
                events = []
        proposals.append(
            {
                **propose_action_window(
                    item_id=item_id, reactions=reactions,
                    clip_window=window, audio_events=events,
                ),
                "uid": uid,
                "clip_idx": int(clip_idx),
                "evidence_tier": posteriors.get("evidence_tier"),
                "posteriors": {
                    key: value for key, value in posteriors.items()
                    if key != "evidence_tier"
                },
            }
        )
    with out_proposals.open("x") as handle:
        for row in proposals:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    sample = review_sample(proposals, args.review_sample, args.seed)
    with out_sample.open("x") as handle:
        for row in sample:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    by_status: dict[str, int] = {}
    snapped = 0
    for row in proposals:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
        snapped += bool(row.get("audio_snap", {}).get("used"))
    summary = {
        "proposer_version": PROPOSER_VERSION,
        **counts,
        "proposals": len(proposals),
        "by_status": by_status,
        "audio_snapped": snapped,
        "review_sample": len(sample),
        "media_cut": False,
        "policy": "proposal_only_exact_artifact_review_required_before_splice",
    }
    out_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
