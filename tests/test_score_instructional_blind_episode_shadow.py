from scripts.score_instructional_blind_episode_shadow import RULE_ID, score_row


def test_shadow_score_uses_frozen_relaxed_rule_without_acceptance() -> None:
    row = {
        "item_id": "instructional:u:0",
        "uid": "u",
        "model": "qwen",
        "rubric": "blind_episode_v1",
        "error": None,
        "result": {
            "episode_pass": False,
            "visual_context": "roleplay_episode",
            "participant_grounding": "affected_party_present",
            "physical_social_action": "yes",
            "quote_function": "sample_or_hypothetical",
        },
    }
    scored = score_row(row)
    assert scored["rule_id"] == RULE_ID
    assert scored["candidate_for_review"] is True
    assert scored["strict_episode_pass"] is False
    assert scored["automatic_acceptance"] is False
    assert scored["corpus_mutation_authorized"] is False
    assert scored["allowed_uses"] == []
    assert scored["transfer_status"] == "failed_transfer"


def test_error_is_retained_but_never_routed() -> None:
    scored = score_row(
        {"item_id": "instructional:u:0", "error": "timeout", "result": None}
    )
    assert scored["candidate_for_review"] is False
    assert scored["source_error"] == "timeout"
