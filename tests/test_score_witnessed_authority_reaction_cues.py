from __future__ import annotations

import json

from scripts.score_witnessed_authority_reaction_cues import (
    score_metadata,
    score_reaction,
)


def test_explicit_arrest_phrase_is_authority_cue():
    result = score_reaction(
        {
            "tag": "declaration",
            "norm": "reckless driving",
            "phrase": "You're under arrest as well",
        }
    )
    assert result["authority_reaction_cue"]
    assert "phrase:under_or_threatened_arrest" in result["cue_ids"]


def test_audited_police_command_is_authority_cue():
    result = score_reaction(
        {
            "tag": "command",
            "norm": "obey police orders",
            "phrase": "Stay where you are!",
            "context": "unlock the window and put it down",
        }
    )
    assert result["authority_reaction_cue"]
    assert (
        "conjunction:command+norm:obey police orders"
        in result["cue_ids"]
    )
    assert "phrase:stay_where_you_are" in result["cue_ids"]


def test_organic_bystander_disapproval_has_no_authority_cue():
    result = score_reaction(
        {
            "tag": "disapproval",
            "norm": "reckless driving",
            "phrase": "What an idiot",
            "context": "Did he get you?",
        }
    )
    assert not result["authority_reaction_cue"]
    assert result["cue_ids"] == []


def test_generic_stop_does_not_trigger_without_authority_context():
    assert not score_reaction(
        {
            "tag": "concern",
            "norm": "non-violence",
            "phrase": "Stop!",
        }
    )["authority_reaction_cue"]


def test_descriptive_on_the_floor_phrase_does_not_trigger():
    assert not score_reaction(
        {
            "tag": "taunt",
            "norm": "mocking",
            "phrase": "Your face is on the floor",
        }
    )["authority_reaction_cue"]


def test_nearby_context_cannot_supply_authority_phrase():
    assert not score_reaction(
        {
            "tag": "bystander objection",
            "norm": "excessive force",
            "phrase": "Why did you hit him?",
            "context": "the officer put his hands behind his back",
        }
    )["authority_reaction_cue"]


def test_generic_command_requires_enforcement_specific_norm():
    assert not score_reaction(
        {
            "tag": "command",
            "norm": "gentle play",
            "phrase": "Don't bite him",
        }
    )["authority_reaction_cue"]


def test_exact_hands_and_weapon_commands_trigger():
    hands = score_reaction(
        {
            "tag": "command",
            "norm": "suspicion",
            "phrase": "Get your hands out of your pockets",
        }
    )
    weapon = score_reaction(
        {
            "tag": "command",
            "norm": "safety",
            "phrase": "Drop the gun!",
        }
    )
    assert "phrase:hands_command" in hands["cue_ids"]
    assert "phrase:drop_weapon" in weapon["cue_ids"]


def test_participant_refusal_to_go_to_jail_does_not_trigger():
    assert not score_reaction(
        {
            "tag": "defiance",
            "norm": "freedom",
            "phrase": "I'm not going to jail",
        }
    )["authority_reaction_cue"]


def test_authority_threat_of_jail_still_triggers():
    result = score_reaction(
        {
            "tag": "warning",
            "norm": "authority",
            "phrase": "You're going to go to jail for misuse of 911",
        }
    )
    assert "phrase:jail_or_citation" in result["cue_ids"]


def test_metadata_scoring_preserves_scene_candidate(tmp_path):
    folder = tmp_path / "dailymotion__example"
    folder.mkdir()
    metadata = {
        "video_id": "dailymotion__example",
        "n_clips": 2,
        "provenance": {"scene": {"reactor_role": "camera_person"}},
        "reactions": [
            {
                "clip_idx": 0,
                "tag": "arrest",
                "norm": "law enforcement",
                "phrase": "You are under arrest",
            },
            {
                "clip_idx": 1,
                "tag": "disapproval",
                "norm": "reckless driving",
                "phrase": "What an idiot",
            },
        ],
    }
    path = folder / "metadata.json"
    path.write_text(json.dumps(metadata))
    rows = score_metadata(path)
    assert rows[0]["strict_witnessed_text_cue_disposition"] == (
        "exclude_strict_preserve_scene_candidate"
    )
    assert rows[0]["corpus_disposition"] is None
    assert rows[1]["strict_witnessed_text_cue_disposition"] == "no_authority_cue"
