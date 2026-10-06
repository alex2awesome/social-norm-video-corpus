from scripts.evaluate_witnessed_reaction_atomic_candidates import (
    disagreement_rows,
    evaluate,
    manual_atoms,
    model_atoms,
)


def manual(candidate_id, role, response, disposition):
    return {
        "candidate_id": candidate_id,
        "visual_identity_role": role,
        "visual_third_party_response": response,
        "speaker_proxy_disposition": disposition,
    }


def result(candidate_id, *, role="separate_bystander", trigger="interpersonal_treatment", staging="organic"):
    return {
        "candidate_id": candidate_id,
        "error": None,
        "result": {
            "reaction_grounded": "yes",
            "response_targets_action": "yes",
            "responder_role": role,
            "trigger_kind": trigger,
            "response_content": "protective_intervention",
            "staging": staging,
        },
    }


def test_manual_atoms_separate_social_trigger_from_bystander_identity():
    safety = manual("a", "generic_safety_bystander", "yes", "failed_social_trigger")
    assert manual_atoms(safety) == {
        "response_present": True,
        "bystander_identity": True,
        "social_trigger": False,
        "witnessed_review": False,
        "instructional_reroute": False,
    }


def test_manual_atoms_routes_staged_bystander_to_instructional():
    staged = manual("a", "bystander", "yes", "instructional_reroute_only")
    atoms = manual_atoms(staged)
    assert atoms["social_trigger"] is True
    assert atoms["witnessed_review"] is False
    assert atoms["instructional_reroute"] is True


def test_model_atoms_keep_staging_out_of_reaction_presence():
    staged = model_atoms(result("a", staging="clearly_staged"))
    assert staged["response_present"] is True
    assert staged["witnessed_review"] is False
    assert staged["instructional_reroute"] is True


def test_visual_v2_abstains_on_unasked_semantic_atoms():
    row = {
        "candidate_id": "a",
        "error": None,
        "result": {
            "visible_response_at_candidate_moment": "yes",
            "visible_response_targets_action": "yes",
            "visual_responder_role": "bystander",
        },
    }
    atoms = model_atoms(row)
    assert atoms["response_present"] is True
    assert atoms["bystander_identity"] is True
    assert atoms["social_trigger"] is None
    assert atoms["witnessed_review"] is None


def test_evaluate_reports_observed_and_fail_closed_missingness():
    ledger = [
        manual("a", "bystander", "yes", "candidate_generation_only"),
        manual("b", "affected_target", "no", "failed_as_role_filter"),
    ]
    model = [result("a"), {"candidate_id": "b", "error": "bad", "result": None}]
    report = evaluate(ledger, {"m": model})
    reaction = report["models"]["m"]["response_present"]
    assert reaction["coverage"] == 0.5
    assert reaction["observed_only"]["recall"] == 1.0
    assert reaction["fail_closed"]["items"] == 2
    assert reaction["fail_closed"]["tp_candidate_ids"] == ["a"]


def test_consensus_requires_every_model_to_vote_true():
    ledger = [manual("a", "bystander", "yes", "candidate_generation_only")]
    first = [result("a")]
    second = [result("a", role="affected_target")]
    report = evaluate(ledger, {"first": first, "second": second})
    strict = report["models"]["strict_model_consensus"]["witnessed_review"]
    assert strict["fail_closed"]["selected"] == 0


def test_disagreement_rows_preserve_manual_note_and_model_evidence():
    ledger = [manual("a", "affected_target", "no", "failed_as_role_filter")]
    ledger[0].update({"storyboard_index": "0", "manual_note": "involved target"})
    prediction = result("a")
    prediction["result"]["evidence"] = "third person reacts"
    rows = disagreement_rows(ledger, {"m": [prediction]})
    assert rows[0]["model_evidence"] == "third person reacts"
    assert rows[0]["manual_note"] == "involved target"
    assert "bystander_identity" in rows[0]["mismatch_atoms"]
