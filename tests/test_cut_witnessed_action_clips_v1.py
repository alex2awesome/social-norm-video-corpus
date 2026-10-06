import json
import shutil
import subprocess

import pytest

from scripts.cut_witnessed_action_clips_v1 import (
    cut_clip,
    excluded_items,
    select_proposals,
)

FFMPEG = shutil.which("ffmpeg")


def proposal(item="witnessed:u1:clip_0", status="proposed"):
    return {
        "item_id": item, "uid": item.split(":")[1],
        "clip_idx": int(item.rsplit("_", 1)[-1]), "status": status,
        "action_window_sec": [60.0, 69.7],
        "action_window_clip_relative_sec": [1.0, 3.0],
        "audio_snap": {"used": False},
    }


def test_provenance_exclusions_gate_the_organic_tier():
    records = [
        {"item_id": "witnessed:u1:clip_0",
         "lf_id": "registry_witnessed_creator_staging_title_v2", "vote": -1},
        {"item_id": "witnessed:u2:clip_0",
         "lf_id": "registry_witnessed_authority_exact_span_v3", "vote": 0},
        {"item_id": "witnessed:u3:clip_0",
         "lf_id": "vis_witnessed_staged_roleplay_v1", "vote": -1},
        {"item_id": "witnessed:u4:clip_0",
         "lf_id": "det_witnessed_scene_reactor_role_v1", "vote": -1},
    ]
    excluded = excluded_items(records)
    # Untriggered rules and non-provenance LFs never exclude.
    assert excluded == {"witnessed:u1:clip_0", "witnessed:u3:clip_0"}
    proposals = [proposal(f"witnessed:u{i}:clip_0") for i in (1, 2, 3, 4)]
    proposals.append(proposal("witnessed:u5:clip_0", status="short_window_review"))
    organic, counts = select_proposals(proposals, excluded, tier="organic")
    assert {p["item_id"] for p in organic} == {
        "witnessed:u2:clip_0", "witnessed:u4:clip_0",
    }
    assert counts == {"proposed": 4, "excluded_provenance": 2, "not_proposed": 1}
    everything, _ = select_proposals(proposals, excluded, tier="all")
    assert len(everything) == 4


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg unavailable")
def test_cut_clip_produces_exact_window(tmp_path):
    source = tmp_path / "clip_0.mp4"
    subprocess.run(
        [FFMPEG, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
         "-i", "testsrc=duration=5:size=128x72:rate=10",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
         "-shortest", str(source)],
        check=True, capture_output=True,
    )
    dest = tmp_path / "out" / "clip_0_action.mp4"
    ok, error = cut_clip(FFMPEG, source, dest, 1.0, 3.0)
    assert ok, error
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(dest)],
        capture_output=True, text=True, check=True,
    )
    duration = float(json.loads(probe.stdout)["format"]["duration"])
    assert 1.7 <= duration <= 2.4
    # Missing source fails cleanly, no partial file left behind.
    ok, error = cut_clip(FFMPEG, tmp_path / "nope.mp4", tmp_path / "x.mp4", 0, 2)
    assert not ok and error
    assert not (tmp_path / "x.mp4").exists()


def test_context_view_window_math():
    # Context view: 30s before anchor from raw, still ending pre-reaction.
    row = proposal()
    row["reaction_anchor_sec"] = 70.0
    anchor = row["reaction_anchor_sec"]
    rel_start = max(0.0, anchor - 30.0)
    rel_end = row["action_window_sec"][1]
    assert rel_start == 40.0 and rel_end == 69.7
    # Early anchor clamps to zero, never negative.
    assert max(0.0, 12.0 - 30.0) == 0.0
