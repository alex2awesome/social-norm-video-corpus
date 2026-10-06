import json
from pathlib import Path

from scripts.run_corpus_lf_snorkel_v1 import (
    export_ranked_candidates,
    fit_and_export,
    materialize,
)
from scripts.weak_signal_registry import load_registry

REGISTRY = load_registry(Path("config/audited_weak_signals_v1.json"))


def make_corpus(root: Path, n_witnessed=6):
    (root / "data" / "transcripts").mkdir(parents=True)
    for i in range(n_witnessed):
        uid = f"reddit__w{i}"
        hit = root / "data" / "hits" / uid
        hit.mkdir(parents=True)
        good = i % 2 == 0
        (hit / "metadata.json").write_text(json.dumps({
            "video_id": uid,
            "title": "queue cutting at the store" if good else "kissing random girls prank",
            "reactions": [{
                "clip_idx": 0, "clip_window": [10.0, 24.0], "start": 20.0,
                "phrase": "stop, that's so rude" if good else "oh wow omg",
                "matched_text": "stop, that's so rude" if good else "oh wow omg",
                "tag": "censure" if good else "exclamation",
            }],
            "provenance": {"scene": {
                "scene_type": "action" if good else "narration",
                "reactor_role": "bystander" if good else "camera_person",
                "severity": 5 if good else 2, "reaction_strength": 5 if good else 2,
            }},
            "audio_events": {"scream_peak": 0.4 if good else 0.01},
        }))
        (hit / "clip_0.mp4").write_bytes(b"x")
        (root / "data" / "transcripts" / f"{uid}.json").write_text(json.dumps({
            "segments": [{"start": 18.0, "end": 21.0,
                          "text": "hey stop, that's so rude" if good else "oh wow omg"}],
        }))
    instr = root / "data" / "instructional" / "yt__i0"
    instr.mkdir(parents=True)
    (instr / "metadata.json").write_text(json.dumps({
        "demos": [
            {"polarity": "violation", "norm": "no interrupting",
             "start_quote": "watch this scenario", "start": 5.0, "end": 15.0},
            {"polarity": "explanation", "norm": "respect"},
        ],
    }))
    (instr / "demo_0.mp4").write_bytes(b"x")
    (root / "data" / "discussion").mkdir(parents=True)
    (root / "data" / "discussion" / "dm__c0.json").write_text(json.dumps({
        "statements": [{"quote": "this happened yesterday and the footage shows it"}],
    }))


def test_end_to_end_materialize_fit_and_rank(tmp_path):
    root = tmp_path / "corpus"
    make_corpus(root)
    out = tmp_path / "run"
    counts = materialize(root, out, REGISTRY)
    assert counts["sources"] == 8  # 6 witnessed + 1 instructional + 1 commentary
    assert counts["errors"] == 0
    assert counts["lf_records"] > 0

    # Resume: nothing is reprocessed and shards are not duplicated.
    before = (out / "lf_records.jsonl").read_text()
    counts2 = materialize(root, out, REGISTRY)
    assert counts2["sources"] == 0 and counts2["skipped"] == 8
    assert (out / "lf_records.jsonl").read_text() == before

    fitted = fit_and_export(out, high=0.7, low=0.3)
    assert fitted["summaries"]["witnessed:norm_event_supported"]["rows"] == 6
    assert fitted["summaries"]["instructional:demonstration_present"]["rows"] == 2
    assert fitted["summaries"]["commentary:occurred_event_supported"]["rows"] == 1

    ranked = export_ranked_candidates(
        out, root, fitted["posteriors_by_item"], high=0.7
    )
    rows = [
        json.loads(line)
        for line in (out / "ranked_norm_violation_candidates.jsonl").read_text().splitlines()
    ]
    assert ranked["ranked_candidates"] == len(rows)
    # Only the clean "good" witnessed items should rank; staged/narration must not.
    good_uids = {f"reddit__w{i}" for i in range(6) if i % 2 == 0}
    assert rows, "expected at least one high-confidence candidate"
    assert {row["uid"] for row in rows} <= good_uids
    for row in rows:
        assert row["acceptance_label"] is None
        assert row["delete_media"] is False
        assert row["posterior_norm_event_supported"] >= 0.7
    # Ordered by review priority.
    scores = [row["review_priority_score"] for row in rows]
    assert scores == sorted(scores, reverse=True)


def test_visual_index_augments_all_pillars(tmp_path):
    root = tmp_path / "corpus"
    make_corpus(root, n_witnessed=2)
    visual_index = {
        ("witnessed", "reddit__w0", 0): {
            "keypoints": {"person_count_mean": 4.0, "person_present_fraction": 0.9},
            "clip_scores": {"mean_probabilities": [0.05, 0.6, 0.1, 0.05, 0.02, 0.08]},
        },
        ("instructional", "yt__i0", 0): {
            "keypoints": {"multiple_people_fraction": 0.8, "person_present_fraction": 0.9},
            "clip_scores": {"mean_probabilities": [0.6, 0.2, 0.1, 0.05, 0.02, 0.05]},
        },
        ("commentary", "dm__c0", 0): {
            "low_level": {"face_present_fraction": 0.7, "multiple_faces_fraction": 0.3,
                          "motion_mean": 0.2},
        },
    }
    out = tmp_path / "run"
    materialize(root, out, REGISTRY, visual_index=visual_index)
    records = [json.loads(line) for line in (out / "lf_records.jsonl").read_text().splitlines()]
    lf_ids_by_item = {}
    for row in records:
        lf_ids_by_item.setdefault(row["item_id"], set()).add(row["lf_id"])
    assert "vis_witnessed_person_count_v1" in lf_ids_by_item["witnessed:reddit__w0:clip_0"]
    assert "vis_witnessed_person_count_v1" not in lf_ids_by_item["witnessed:reddit__w1:clip_0"]
    assert "vis_instructional_demo_scene_v1" in lf_ids_by_item["instructional:yt__i0:demo_0"]
    assert "vis_commentary_scene_activity_v1" in lf_ids_by_item["commentary:dm__c0:stmt_0"]
    fitted = fit_and_export(out, high=0.7, low=0.3)
    assert fitted["summaries"]["commentary:event_present_in_source"]["rows"] == 1


def test_materialize_survives_corrupt_metadata(tmp_path):
    root = tmp_path / "corpus"
    make_corpus(root, n_witnessed=2)
    bad = root / "data" / "hits" / "reddit__bad"
    bad.mkdir(parents=True)
    (bad / "metadata.json").write_text("{not json")
    out = tmp_path / "run"
    counts = materialize(root, out, REGISTRY)
    assert counts["errors"] == 1
    assert counts["sources"] == 4
    errors = [
        json.loads(line)
        for line in (out / "materialize_errors.jsonl").read_text().splitlines()
    ]
    assert errors[0]["source_key"] == "witnessed:reddit__bad"
