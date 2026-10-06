import pytest

from scripts.build_commentary_hierarchical_source_packets_v2 import (
    build_packets,
    grounded_quote_span,
    visual_deixis_anchors,
)


def label(uid="dailymotion__a"):
    return {
        "uid": uid,
        "item_id": f"commentary:{uid}:0",
        "text_label_manual_reviewed": True,
        "result": {
            "decision": "accept",
            "social_actor_grounded": "yes",
            "concrete_behavior": "yes",
            "target_or_shared_context_grounded": "yes",
            "normative_stance_grounded": "yes",
            "normalized_behavior": "a customer hits a worker",
            "normalized_norm": "do not assault service workers",
            "behavior_evidence_quote": "the customer struck the worker",
            "stance_evidence_quote": "that conduct was unacceptable",
        },
    }


def transcript(uid="dailymotion__a"):
    return {
        "uid": uid,
        "segments": [
            {"start": 4, "end": 7, "text": "The video shows what happened."},
            {"start": 20, "end": 23, "text": "The customer struck"},
            {"start": 23, "end": 25, "text": "the worker yesterday."},
            {"start": 26, "end": 29, "text": "That conduct was unacceptable."},
        ],
    }


def metadata(uid="dailymotion__a"):
    return {"uid": uid, "title": "Customer attacks worker", "query_source": "taxonomy"}


def video(uid="dailymotion__a"):
    return {"uid": uid, "source_path": f"/video/{uid}.mp4", "duration_sec": 60}


def test_quote_grounding_can_cross_adjacent_transcript_segments():
    segments = transcript()["segments"]
    assert grounded_quote_span("the customer struck the worker", segments) == (20.0, 25.0)


def test_visual_deixis_creates_bounded_candidate_anchor():
    anchors = visual_deixis_anchors(transcript()["segments"], 60)
    assert anchors == [{
        "kind": "visual_deixis",
        "start_sec": 0.0,
        "end_sec": 15.0,
        "evidence": "The video shows what happened.",
    }]


def test_packet_uses_script_behavior_but_refuses_visual_certification():
    packets = build_packets([label()], [metadata()], [transcript()], [video()])
    row = packets[0]
    assert row["action_label"] == "a customer hits a worker"
    assert row["behavior_quote_span_sec"] == [20.0, 25.0]
    assert row["stance_quote_span_sec"] == [26.0, 29.0]
    assert {anchor["kind"] for anchor in row["anchors"]} == {
        "commentary_statement", "visual_deixis"
    }
    assert row["script_certifies_visual_event"] is False
    assert "visual" not in row["policy"].split("_")[0]


def test_prior_vlm_bounds_are_preserved_only_as_search_anchors():
    packets = build_packets(
        [label()], [metadata()], [transcript()], [video()],
        [{"uid": "dailymotion__a", "bounds": [{
            "start_sec": 30, "end_sec": 36, "model": "qwen"
        }]}],
    )
    prior = [a for a in packets[0]["anchors"] if a["kind"] == "prior_vlm"]
    assert prior == [{
        "kind": "prior_vlm", "start_sec": 30.0, "end_sec": 36.0,
        "evidence": "qwen",
    }]


def test_unreviewed_or_ungrounded_text_label_fails_closed():
    row = label()
    row["text_label_manual_reviewed"] = False
    with pytest.raises(ValueError, match="manual review"):
        build_packets([row], [metadata()], [transcript()], [video()])
    row = label()
    row["result"]["target_or_shared_context_grounded"] = "no"
    with pytest.raises(ValueError, match="target_or_shared_context"):
        build_packets([row], [metadata()], [transcript()], [video()])


def test_quotes_must_be_in_timestamped_transcript_not_title():
    row = label()
    row["result"]["behavior_evidence_quote"] = "customer attacks worker"
    with pytest.raises(ValueError, match="not grounded"):
        build_packets([row], [metadata()], [transcript()], [video()])


def test_every_selected_uid_requires_metadata_transcript_and_video():
    with pytest.raises(ValueError, match="metadata is missing"):
        build_packets([label()], [], [transcript()], [video()])
