from pathlib import Path

from scripts.labeling_functions_v1 import validate_lf_record
from scripts.materialize_corpus_lf_records_v1 import (
    commentary_item_records,
    instructional_item_records,
    witnessed_item_records,
)
from scripts.weak_signal_registry import load_registry

REGISTRY = load_registry(Path("config/audited_weak_signals_v1.json"))


def witnessed_metadata():
    return {
        "video_id": "reddit__abc",
        "title": "Guy cuts the whole queue at the airport",
        "channel": "some_channel",
        "reactions": [
            {
                "clip_idx": 0,
                "clip_window": [60.0, 74.0],
                "start": 70.0,
                "end": 71.5,
                "phrase": "that's so rude",
                "matched_text": "hey, stop, that's so rude",
                "tag": "censure",
                "norm": "queueing",
            }
        ],
        "provenance": {
            "scene": {
                "scene_type": "action",
                "reactor_role": "bystander",
                "violator_role": "subject",
                "severity": 4,
                "reaction_strength": 4,
            }
        },
        "audio_events": {"scream_peak": 0.31, "commotion_peak": 0.05, "n_events": 2},
    }


def transcript():
    return [
        {"start": 58.0, "end": 62.0, "text": "he just walked past everyone"},
        {"start": 69.5, "end": 72.0, "text": "hey, stop, that's so rude"},
    ]


def by_lf(records):
    return {(r["lf_id"], r["target"]): r for r in records}


def test_witnessed_records_cover_registry_and_deterministic_lfs():
    records, gates = witnessed_item_records(
        "reddit__abc", witnessed_metadata(), transcript(), REGISTRY,
        clip_exists={0: True},
    )
    for record in records:
        validate_lf_record(record)
    lfs = by_lf(records)
    # Intervention scan finds "stop"/"rude" -> +1 reaction_grounded.
    scan = lfs[("registry_witnessed_clipwide_intervention_scan_v1", "reaction_grounded")]
    assert scan["vote"] == 1
    # No authority command, no staging title, no WWYD channel -> abstain.
    assert lfs[("registry_witnessed_authority_exact_span_v3", "independent_bystander_signal")]["vote"] == 0
    assert lfs[("registry_witnessed_creator_staging_title_v2", "independent_bystander_signal")]["vote"] == 0
    # Scene roles: bystander -> +1 subtype; action scene -> +1 norm event.
    assert lfs[("det_witnessed_scene_reactor_role_v1", "independent_bystander_signal")]["vote"] == 1
    assert lfs[("det_witnessed_scene_action_type_v1", "norm_event_supported")]["vote"] == 1
    # Targeted objection language -> +1 on both targets.
    assert lfs[("det_witnessed_reaction_content_norm_event_supported_v1", "norm_event_supported")]["vote"] == 1
    assert lfs[("det_witnessed_reaction_content_reaction_grounded_v1", "reaction_grounded")]["vote"] == 1
    # Audio peak 0.31 >= 0.2 -> +1.
    assert lfs[("det_witnessed_audio_event_peak_v1", "reaction_grounded")]["vote"] == 1
    (gate,) = gates
    assert gate["item_id"] == "witnessed:reddit__abc:clip_0"
    assert gate["eligible"] is True and not gate["failed_gates"]


def test_witnessed_negative_and_abstain_paths():
    metadata = witnessed_metadata()
    metadata["title"] = "kissing random girls prank (social experiment)"
    metadata["provenance"]["scene"].update(reactor_role="victim", scene_type="narration")
    metadata["reactions"][0].update(
        phrase="oh wow omg", matched_text="oh wow omg", tag="exclamation"
    )
    metadata["audio_events"] = {}
    quiet = [{"start": 60.0, "end": 62.0, "text": "oh wow omg"}]
    records, gates = witnessed_item_records(
        "reddit__abc", metadata, quiet, REGISTRY, clip_exists={0: False}
    )
    lfs = by_lf(records)
    assert lfs[("registry_witnessed_creator_staging_title_v2", "independent_bystander_signal")]["vote"] == -1
    assert lfs[("det_witnessed_scene_reactor_role_v1", "independent_bystander_signal")]["vote"] == -1
    assert lfs[("det_witnessed_scene_action_type_v1", "norm_event_supported")]["vote"] == -1
    # Generic affect only -> -1 reaction content; scan abstains (no mechanism).
    assert lfs[("det_witnessed_reaction_content_reaction_grounded_v1", "reaction_grounded")]["vote"] == -1
    assert lfs[("registry_witnessed_clipwide_intervention_scan_v1", "reaction_grounded")]["vote"] == 0
    assert lfs[("det_witnessed_audio_event_peak_v1", "reaction_grounded")]["vote"] == 0
    (gate,) = gates
    assert gate["failed_gates"] == ["media_decodes"]


def test_witnessed_authority_command_votes_against_subtype():
    metadata = witnessed_metadata()
    metadata["reactions"][0].update(
        phrase="get on the ground", matched_text="get on the ground, show me your hands",
        tag="command",
    )
    records, _ = witnessed_item_records(
        "reddit__abc", metadata, transcript(), REGISTRY, clip_exists={0: True}
    )
    lfs = by_lf(records)
    row = lfs[("registry_witnessed_authority_exact_span_v3", "independent_bystander_signal")]
    assert row["vote"] == -1


def test_instructional_records_polarity_and_transition():
    metadata = {
        "demos": [
            {"polarity": "violation", "norm": "do not interrupt",
             "start_quote": "watch this scenario with two coworkers",
             "start": 30.0, "end": 45.0},
            {"polarity": "explanation", "norm": "respect",
             "start_quote": "she said it happened last year"},
        ]
    }
    records, gates = instructional_item_records(
        "yt__def", metadata, REGISTRY, demo_clip_exists={0: True, 1: False}
    )
    for record in records:
        validate_lf_record(record)
    rows = {(r["item_id"], r["lf_id"]): r for r in records}
    demo0 = "instructional:yt__def:demo_0"
    demo1 = "instructional:yt__def:demo_1"
    assert rows[(demo0, "registry_instructional_non_explanation_review_priority_v1")]["vote"] == 1
    assert rows[(demo0, "det_instructional_demo_transition_cue_v1")]["vote"] == 1
    assert rows[(demo1, "registry_instructional_non_explanation_review_priority_v1")]["vote"] == 0
    assert rows[(demo1, "det_instructional_explanation_polarity_v1")]["vote"] == -1
    assert rows[(demo1, "det_instructional_demo_transition_cue_v1")]["vote"] == -1
    gate0, gate1 = gates
    assert gate0["eligible"] is True
    assert "demonstration_interval_present" in gate1["failed_gates"]


def test_commentary_occurred_vs_hypothetical():
    record = {
        "title": "my thoughts on rude drivers",
        "statements": [
            {"quote": "this driver blocked the crosswalk yesterday and people said it was awful"},
            {"quote": "imagine if you just walked into someone's house uninvited"},
            {"quote": "the footage shows him grabbing the cart"},
        ],
    }
    records, gates = commentary_item_records("dm__ghi", record, REGISTRY)
    for row in records:
        validate_lf_record(row)
    rows = {(r["item_id"], r["lf_id"]): r for r in records}
    s0 = "commentary:dm__ghi:stmt_0"
    s1 = "commentary:dm__ghi:stmt_1"
    s2 = "commentary:dm__ghi:stmt_2"
    assert rows[(s0, "det_commentary_occurred_event_language_v2")]["vote"] == 1
    assert rows[(s1, "det_commentary_occurred_event_language_v2")]["vote"] == -1
    assert rows[(s2, "det_commentary_footage_deixis_v2")]["vote"] == 1
    # Non-event title: capture-title rule abstains as untriggered.
    title_row = rows[(s0, "registry_commentary_capture_title_event_v1")]
    assert title_row["vote"] == 0 and title_row["abstain_reason"] == "rule_not_triggered"
    # No dual-VLM score supplied: that rule emits no record at all.
    assert (s0, "registry_commentary_dual_vlm_retrieval_core_v1") not in rows
    assert all(gate["eligible"] for gate in gates)


def test_commentary_transcript_window_recovers_event_language():
    record = {
        "title": "driver caught on camera attacking a cyclist",
        "statements": [
            {"quote": "that is absolutely disgusting behavior", "start": 40.0, "end": 42.0},
        ],
    }
    segments = [
        {"start": 10.0, "end": 12.0, "text": "welcome back to the channel"},
        {"start": 33.0, "end": 37.0,
         "text": "look what happened yesterday, the footage shows him shoving the cyclist"},
        {"start": 90.0, "end": 93.0, "text": "imagine if everyone drove like that"},
    ]
    records, _ = commentary_item_records(
        "dm__win", record, REGISTRY,
        transcript_segments=segments, dual_vlm_positive=True,
    )
    rows = {r["lf_id"]: r for r in records}
    # Stance quote alone has no event language, but the +-20s window does.
    occurred = rows["det_commentary_occurred_event_language_v2"]
    assert occurred["vote"] == 1 and occurred["evidence"]["windowed"] is True
    assert rows["det_commentary_footage_deixis_v2"]["vote"] == 1
    # The far-away "imagine" line is outside the window and outside the quote.
    assert occurred["evidence"]["hypothetical_language"] is False
    # Audited registry votes: capture-title and dual-VLM on event_present_in_source.
    title_row = rows["registry_commentary_capture_title_event_v1"]
    assert title_row["vote"] == 1
    assert title_row["target"] == "event_present_in_source"
    vlm_row = rows["registry_commentary_dual_vlm_retrieval_core_v1"]
    assert vlm_row["vote"] == 1 and vlm_row["confidence"] == 1.0


def test_commentary_animal_title_does_not_trigger_capture_cue():
    record = {
        "title": "dog attack caught on camera",
        "statements": [{"quote": "that was wild"}],
    }
    records, _ = commentary_item_records("dm__dog", record, REGISTRY)
    rows = {r["lf_id"]: r for r in records}
    assert rows["registry_commentary_capture_title_event_v1"]["vote"] == 0
