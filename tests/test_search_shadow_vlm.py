import json
from pathlib import Path

import pytest

from scripts.evaluate_search_shadow_ollama import metrics, score
from scripts.evaluate_instructional_score_bands import assign_band
from scripts.export_search_shadow_vlm_benchmark import build_records
from scripts.run_ollama_sheet_audit import (
    successful_prediction_keys,
    validate_result,
)
from scripts.run_ollama_transcript_audit import (
    SYSTEMS,
    successful_prediction_keys as successful_transcript_prediction_keys,
    transcript_text,
    validate_result as validate_transcript_result,
)
from scripts.run_open_vlm_scene_benchmark import (
    parse_json as parse_video_json,
    successful_prediction_keys as successful_video_prediction_keys,
)


def test_build_records_joins_by_ordinal_and_derives_gold(tmp_path: Path):
    sheet = tmp_path / "sheet.jpg"
    sheet.write_bytes(b"sheet")
    manifest = {
        "records": [
            {
                "ordinal": 0,
                "uid": "dailymotion__x",
                "pillar": "instructional",
                "query": "role play",
                "artifact_status": "rendered",
                "sheet_path": "sheet.jpg",
                "sheet_sha256": "abc",
            }
        ]
    }
    reviews = [
        {
            "audit_index": 0,
            "uid": "dailymotion__x",
            "intended_pillar": "instructional",
            "visual_scene": "yes",
            "decision": "accept_after_localization",
            "strict_target_pass": True,
        }
    ]

    records = build_records(manifest, reviews, tmp_path)

    assert records[0]["gold_social_behavior_scene"] is True
    assert records[0]["gold_target_source_pass"] is True


def test_score_and_metrics_are_conservative_for_uncertain():
    assert score("yes", 0.9) == 0.9
    assert score("no", 0.9) == pytest.approx(0.1)
    assert score("uncertain", 0.99) == 0.5
    rows = [
        {"gold": True, "result": {"label": "yes", "confidence": 0.9}},
        {"gold": False, "result": {"label": "uncertain", "confidence": 0.9}},
    ]

    result = metrics(rows, "gold", "label", 0.8)

    assert result == {
        "n": 2,
        "threshold": 0.8,
        "tp": 1,
        "fp": 0,
        "tn": 1,
        "fn": 0,
        "precision": 1.0,
        "recall": 1.0,
    }


def test_contact_sheet_result_enforces_pillar_invariants():
    result = {
        "social_behavior_scene_visible": "no",
        "target_pillar_signal_visible": "uncertain",
        "observable_action_visible": "no",
        "interaction_or_shared_context_visible": "no",
        "action_then_causal_response_visible": "na",
        "commentary_event_visual_pair_present": "na",
        "presentation_or_context_only": "yes",
        "hypothesized_behavior_directly_visible": "uncertain",
        "behavior_intentionally_demonstrated_or_exemplified": "uncertain",
        "medium": "graphics",
        "confidence": 0.95,
        "evidence": "Only a title card is visible.",
    }

    validate_result(result, {"pillar": "instructional"}, "blind")


def test_contact_sheet_resume_retries_failures():
    base = {
        "item_id": "instructional:x:0",
        "mode": "conditioned",
        "model": "qwen",
        "rubric_version": "contact_sheet_v4",
    }

    assert not successful_prediction_keys(
        [{**base, "result": None, "error": "schema failure"}]
    )
    assert successful_prediction_keys(
        [{**base, "result": {"ok": True}, "error": None}]
    ) == {
        (
            "instructional:x:0",
            "conditioned",
            "qwen",
            "contact_sheet_v4",
        )
    }


def test_video_resume_is_rubric_specific_and_retries_failures():
    base = {
        "item_id": "instructional:x:0",
        "mode": "conditioned",
        "model": "qwen",
        "rubric": "v3",
    }

    assert not successful_video_prediction_keys(
        [{**base, "result": None, "error": "context failure"}]
    )
    assert successful_video_prediction_keys(
        [{**base, "result": {"ok": True}, "error": None}]
    ) == {("instructional:x:0", "conditioned", "qwen", "v3")}


def test_v4_video_rubric_requires_social_domain_and_relabel_separation():
    result = {
        "social_norm_domain": "no",
        "situated_social_scenario_visible": "yes",
        "observable_social_behavior_or_speech": "no",
        "usable_demo_after_relabel": "no",
        "proposed_norm_supported": "no",
        "procedural_or_nonsocial_activity_only": "yes",
        "presentation_or_context_only": "no",
        "instructional_demo_present": "no",
        "witnessed_action_then_reaction_organic": "na",
        "commentary_event_footage_present": "na",
        "localization_quality": "clean",
        "depiction_type": "enacted_scene",
        "evidence": "Two people demonstrate a self-defense technique.",
    }

    assert parse_video_json(json.dumps(result), "v4") == result


def test_transcript_resume_retries_failures():
    base = {
        "item_id": "instructional:x:0",
        "model": "qwen",
        "rubric_version": "transcript_gate_v2",
    }

    assert not successful_transcript_prediction_keys(
        [{**base, "result": None, "error": "schema failure"}]
    )
    assert successful_transcript_prediction_keys(
        [{**base, "result": {"ok": True}, "error": None}]
    ) == {
        (
            "instructional:x:0",
            "qwen",
            "transcript_gate_v2",
        )
    }


def test_transcript_gate_preserves_timestamps_and_denies_visual_certainty():
    text = transcript_text(
        {"segments": [{"start": 1.25, "end": 2.5, "text": " Take turns. "}]}
    )
    assert text == "[1.25-2.50] Take turns."
    result = {
        "social_norm_topic_supported": "yes",
        "specific_hypothesis_supported": "yes",
        "demonstration_or_example_language_present": "yes",
        "situated_dialogue_or_action_cues_present": "yes",
        "discussion_or_advice_only": "no",
        "visual_depiction_determined": "no_transcript_cannot_determine_pixels",
        "confidence": 0.9,
        "candidate_windows": [],
        "evidence": "The transcript contains a scenario.",
    }
    validate_transcript_result(result)


def test_transcript_rubrics_are_frozen_and_versioned():
    assert set(SYSTEMS) == {"transcript_gate_v1", "transcript_gate_v2"}
    assert "CORE SOCIAL-NORM BEHAVIOR FAMILY" in SYSTEMS["transcript_gate_v2"]
    assert SYSTEMS["transcript_gate_v1"] != SYSTEMS["transcript_gate_v2"]


def test_instructional_band_requires_independent_audiovisual_agreement():
    visual = {
        "result": {
            "target_pillar_signal_visible": "yes",
            "social_behavior_scene_visible": "yes",
        }
    }
    transcript = {
        "result": {
            "specific_hypothesis_supported": "yes",
            "demonstration_or_example_language_present": "yes",
            "social_norm_topic_supported": "yes",
            "situated_dialogue_or_action_cues_present": "yes",
        }
    }
    assert assign_band(visual, transcript) == "audiovisual_agreement"
    transcript["result"]["specific_hypothesis_supported"] = "no"
    assert assign_band(visual, transcript) == "semantic_conflict_review"


def test_instructional_rescue_is_review_only():
    visual = {
        "result": {
            "target_pillar_signal_visible": "no",
            "social_behavior_scene_visible": "yes",
        }
    }
    transcript = {
        "result": {
            "specific_hypothesis_supported": "no",
            "demonstration_or_example_language_present": "yes",
            "social_norm_topic_supported": "yes",
            "situated_dialogue_or_action_cues_present": "yes",
        }
    }
    assert assign_band(visual, transcript) == "localized_rescue_review"
