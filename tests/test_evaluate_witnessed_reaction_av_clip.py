from scripts.evaluate_witnessed_reaction_av_clip import candidate_positive, evaluate


def manual(item_id, role="bystander", expectation="interpersonal_treatment"):
    return {
        "item_id": item_id,
        "reaction_source_role": role,
        "reaction_grounding": "visible_on_scene",
        "reaction_content": "targeted_objection",
        "temporal_relation": "action_established_before_reaction",
        "expectation_kind": expectation,
    }


def candidate(item_id, role="separate_bystander", trigger="interpersonal_treatment"):
    return {
        "item_id": item_id,
        "candidate_id": item_id + ":0",
        "error": None,
        "result": {
            "reaction_grounded": "yes",
            "action_before_or_overlaps_response": "yes",
            "response_targets_action": "yes",
            "responder_role": role,
            "response_content": "targeted_objection",
            "trigger_kind": trigger,
        },
    }


def test_candidate_positive_keeps_authority_and_social_separate():
    row = candidate("a", role="authority_or_host")["result"]
    assert not candidate_positive(row, include_authority=False, require_social=False)
    assert candidate_positive(row, include_authority=True, require_social=True)


def test_evaluate_fail_closes_clips_without_candidates():
    report = evaluate([manual("a"), manual("b")], [candidate("a")])
    metrics = report["rules"]["strict_bystander_reaction"]["fail_closed"]
    assert metrics["tp"] == 1
    assert metrics["fn"] == 1
    assert report["clips_with_candidates"] == 1


def test_social_route_rejects_safety_even_when_reaction_route_accepts():
    report = evaluate(
        [manual("a", expectation="health_or_physical_safety_only")],
        [candidate("a", trigger="private_physical_safety")],
    )
    assert report["rules"]["strict_bystander_reaction"]["fail_closed"]["tp"] == 1
    assert report["rules"]["strict_bystander_social"]["fail_closed"]["selected"] == 0
