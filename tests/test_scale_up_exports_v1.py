import json

import pytest

from scripts.export_instructional_demo_tier_v1 import export, tier_items
from scripts.propose_commentary_event_windows_v1 import (
    both_high_uids,
    candidate_windows,
)


def posterior_row(item, sem=1, vis=1, failed=(), unknown=()):
    return {
        "item_id": item, "posterior_positive": 0.806,
        "family_votes": {
            k: v for k, v in (("explicit_semantics", sem), ("visual_depiction", vis))
            if v is not None
        },
        "gate": {"failed_gates": list(failed), "unknown_gates": list(unknown)},
        "shadow_band": "insufficient_evidence_abstain",
    }


def test_tier_selection_requires_both_families_and_clean_gates(tmp_path):
    rows = [
        posterior_row("instructional:u1:demo_0"),
        posterior_row("instructional:u2:demo_0", vis=None),
        posterior_row("instructional:u3:demo_0", vis=-1),
        posterior_row("instructional:u4:demo_0", failed=["media_decodes"]),
        posterior_row("instructional:u5:demo_0", unknown=["bounds_valid"]),
    ]
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert [r["item_id"] for r in tier_items(path)] == ["instructional:u1:demo_0"]


def test_export_writes_manifest_with_metadata(tmp_path):
    root = tmp_path / "corpus"
    demo_dir = root / "data" / "instructional" / "u1"
    demo_dir.mkdir(parents=True)
    (demo_dir / "demo_0.mp4").write_bytes(b"video")
    (demo_dir / "metadata.json").write_text(json.dumps({
        "title": "manners 101",
        "demos": [{"norm": "no interrupting", "polarity": "violation"}],
    }))
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [
        posterior_row("instructional:u1:demo_0"),
        posterior_row("instructional:u1:demo_3"),  # no clip file
    ]))
    out = tmp_path / "export"
    summary = export(root, path, out)
    assert summary["exported"] == 1 and summary["missing_clip"] == 1
    row = json.loads((out / "instructional_demo_tier_manifest.jsonl").read_text())
    assert row["norm"] == "no interrupting"
    assert row["acceptance_label"] is None
    assert row["disposition"] == "review_first_candidate"
    with pytest.raises(FileExistsError):
        export(root, path, out)


def test_commentary_window_merge_and_anchor_priority():
    segments = [
        {"start": 100.0, "end": 104.0, "text": "the footage shows him shoving her"},
        {"start": 118.0, "end": 121.0, "text": "this happened yesterday downtown"},
        {"start": 300.0, "end": 304.0, "text": "unrelated chatter"},
    ]
    statements = [{"start": 110.0, "quote": "that is so rude"}]
    windows = candidate_windows(segments, statements)
    # Deixis@100 [95,125], occurred@118 [113,143], statement@110 [90,130] all
    # overlap -> one merged window keeping the strongest anchor (deixis).
    assert len(windows) == 1
    assert windows[0]["anchor"] == "footage_deixis"
    assert windows[0]["start"] == 90.0 and windows[0]["end"] == 143.0


def test_commentary_window_cap_and_clamp():
    segments = [
        {"start": float(1000 * i), "end": 1000.0 * i + 3,
         "text": "you can see the clip shows it"} for i in range(8)
    ]
    windows = candidate_windows(segments, [])
    assert len(windows) == 5
    zero = candidate_windows(
        [{"start": 2.0, "end": 4.0, "text": "video shows the incident"}], []
    )
    assert zero[0]["start"] == 0.0  # clamped, never negative


def test_both_high_uids(tmp_path):
    def write(name, rows):
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows))
    write("posteriors_commentary_occurred_event_supported.jsonl", [
        {"item_id": "commentary:a:stmt_0", "shadow_band": "high_confidence_candidate"},
        {"item_id": "commentary:b:stmt_0", "shadow_band": "high_confidence_candidate"},
    ])
    write("posteriors_commentary_event_present_in_source.jsonl", [
        {"item_id": "commentary:b:stmt_1", "shadow_band": "high_confidence_candidate"},
        {"item_id": "commentary:c:stmt_0", "shadow_band": "insufficient_evidence_abstain"},
    ])
    assert both_high_uids(tmp_path) == {"b"}


def test_commentary_window_geometry_is_tunable():
    segments = [{"start": 100.0, "end": 103.0, "text": "watch this, the footage shows it"}]
    tight = candidate_windows(segments, [], deixis_pre_sec=1.0, deixis_post_sec=18.0)
    assert tight[0]["start"] == 99.0 and tight[0]["end"] == 118.0
    default = candidate_windows(segments, [])
    assert default[0]["start"] == 95.0 and default[0]["end"] == 125.0
