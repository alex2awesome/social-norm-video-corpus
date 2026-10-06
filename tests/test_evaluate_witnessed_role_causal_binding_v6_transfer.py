import copy

import pytest

from scripts.evaluate_witnessed_role_causal_binding_v6_transfer import evaluate


def result(role="separate_bystander", evidence="A third person objects."):
    return {
        "reaction_grounded": "yes",
        "action_before_or_overlaps_response": "yes",
        "response_targets_action": "yes",
        "responder_role": role,
        "response_content": "targeted_objection",
        "trigger_kind": "interpersonal_treatment",
        "staging": "no_clear_staging_evidence",
        "evidence": evidence,
    }


def fixture():
    selection = [{
        "item_id": "witnessed:u:clip_0",
        "uid": "u",
        "cohort": "v3_positive_enrichment",
        "candidates": [{"candidate_id": "c"}],
    }]
    manual = [{
        "candidate_id": "c", "item_id": "witnessed:u:clip_0", "uid": "u",
        **result(), "manual_evidence": "A separate person objects to the act.",
        "manual_note": "",
    }]
    score = [{
        "candidate_id": "c", "item_id": "witnessed:u:clip_0", "uid": "u",
        "model": "m", "prompt_version": "v", "result": result(), "error": None,
    }]
    review = [{
        "audit_index": "0", "item_id": "c", "uid": "u", "model": "m",
        "prompt_version": "v", "structured_prediction_supported": "yes",
        "rationale_supported": "yes", "input_modality_claim_supported": "yes",
        "reaction_decision_supported": "yes",
        "role_binding_supported": "yes", "trigger_binding_supported": "yes",
        "staging_decision_supported": "yes", "unsupported_claim_types": "",
        "manual_rationale": "Supported.",
    }]
    manual[0].update({
        "source_audio_review_status": "reviewed",
        "speaker_identity_basis": "voice_and_visible_turn_binding",
    })
    media = [{"candidate_id": "c", "audio_present": True}]
    return selection, manual, score, review, media


def test_evaluates_clip_level_without_treating_candidate_as_independent():
    selection, manual, score, review, media = fixture()
    report = evaluate(selection, manual, score, score, review, review, media)
    cohort = report["cohorts"]["v3_positive_enrichment"]
    assert cohort["clips"] == 1
    assert cohort["candidates"] == 1
    assert cohort["clip_level"]["v6"]["fail_closed"]["tp"] == 1
    assert report["eligible_for_automatic_acceptance"] is False
    assert report["passes_this_transfer_gate"] is False
    assert report["promotion_gate_checks"][
        "v6_input_modality_matches_audiovisual_claim"
    ] is False
    assert report["posthoc_modality_correction"]["model_did_not_consume"] == [
        "mp4_audio_track"
    ]


def test_requires_exact_model_coverage():
    selection, manual, score, review, media = fixture()
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate(selection, manual, [], score, review, review, media)


def test_requires_review_of_every_rationale():
    selection, manual, score, review, media = fixture()
    bad = copy.deepcopy(review)
    bad[0]["rationale_supported"] = ""
    with pytest.raises(ValueError, match="rationale_supported"):
        evaluate(selection, manual, score, score, review, bad, media)


def test_requires_every_atomic_output_judgment_and_lineage():
    selection, manual, score, review, media = fixture()
    bad = copy.deepcopy(review)
    bad[0]["role_binding_supported"] = ""
    with pytest.raises(ValueError, match="role_binding_supported"):
        evaluate(selection, manual, score, score, review, bad, media)
    bad = copy.deepcopy(review)
    bad[0]["prompt_version"] = "changed"
    with pytest.raises(ValueError, match="lineage mismatch"):
        evaluate(selection, manual, score, score, review, bad, media)
