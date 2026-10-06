from scripts.evaluate_instructional_blind_symbolic_v1 import (
    episode_without_completeness, evaluate, label_with_relaxed_episode,
)


def record(item, result):
    return {"item_id": item, "error": None, "result": result}


def test_evaluate_keeps_episode_and_label_targets_separate() -> None:
    semantics = [{"item_id": "a", "uid": "u", "polarity": "violation"}]
    ledger = [{
        "item_id": "a", "visual_form": "roleplay", "visual_demo": "yes",
        "usable_demo": "yes", "note": "valid",
    }]
    episode = {
        "episode_pass": True,
        "visual_context": "roleplay_episode",
        "participant_grounding": "affected_party_present",
        "physical_social_action": "yes",
        "quote_function": "situated_to_present_party",
    }
    symbol = {
        "label_pass": True,
        "behavior_match": "yes",
        "social_basis": "interpersonal_treatment",
        "relation_matches_proposed_polarity": True,
    }
    rows, summary = evaluate(
        semantics, ledger,
        {"qwen": [record("a", episode)], "gemma": [record("a", episode)]},
        {"qwen": [record("a", symbol)], "gemma": [record("a", symbol)]}, 1,
    )
    assert summary["episode_stage"]["all_model_consensus"]["precision"] == 1
    assert summary["label_stage"]["all_model_consensus"]["recall"] == 1
    assert rows[0]["label_consensus"] is True
    assert summary["automatic_acceptance"] is False


def test_relaxed_episode_drops_only_completeness() -> None:
    episode = {
        "episode_pass": False,
        "visual_context": "roleplay_episode",
        "participant_grounding": "affected_party_present",
        "physical_social_action": "yes",
        "quote_function": "situated_to_present_party",
        "episode_complete": "no",
    }
    symbolic = {
        "label_pass": False,
        "behavior_match": "yes",
        "social_basis": "interpersonal_treatment",
        "relation_matches_proposed_polarity": True,
    }
    assert episode_without_completeness(episode) is True
    assert label_with_relaxed_episode(episode, symbolic) is True


def test_relaxed_episode_still_requires_participant_and_behavior() -> None:
    episode = {
        "visual_context": "connected_episode",
        "participant_grounding": "actor_only",
        "physical_social_action": "no",
        "quote_function": "narration_or_report",
        "episode_complete": "yes",
    }
    assert episode_without_completeness(episode) is False


def test_indexed_visual_and_sparse_semantic_ledgers_join_without_label_leakage() -> None:
    semantics = [
        {"audit_index": 0, "item_id": "a", "uid": "u0"},
        {"audit_index": 1, "item_id": "b", "uid": "u1"},
    ]
    visual_ledger = [
        {"audit_index": "0", "visual_form": "roleplay", "visual_demo": "yes", "note": "scene"},
        {"audit_index": "1", "visual_form": "talking_head", "visual_demo": "no", "note": "lecture"},
    ]
    semantic_ledger = [{"audit_index": "0", "exact_usable": "yes", "note": "match"}]
    episode = {
        "episode_pass": True,
        "visual_context": "roleplay_episode",
        "participant_grounding": "affected_party_present",
        "physical_social_action": "yes",
        "quote_function": "situated_to_present_party",
    }
    symbol = {
        "label_pass": True,
        "behavior_match": "yes",
        "social_basis": "interpersonal_treatment",
        "relation_matches_proposed_polarity": True,
    }
    rows, summary = evaluate(
        semantics,
        visual_ledger,
        {"qwen": [record("a", episode), record("b", episode)]},
        {"qwen": [record("a", symbol), record("b", symbol)]},
        2,
        semantic_ledger,
    )
    assert [row["manual_usable_demo"] for row in rows] == [True, False]
    assert summary["manual_semantic_rows"] == 1
