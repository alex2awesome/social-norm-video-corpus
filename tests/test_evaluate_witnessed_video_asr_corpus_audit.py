import pytest

from scripts.evaluate_witnessed_video_asr_corpus_audit import (
    atomic_values,
    evaluate,
)


def manual(candidate_id, item_id, uid, *, grounded="yes", role="separate_bystander", trigger="interpersonal_treatment"):
    return {
        "candidate_id": candidate_id,
        "item_id": item_id,
        "uid": uid,
        "reaction_grounded": grounded,
        "action_before_or_overlaps_response": "yes",
        "response_targets_action": "yes",
        "responder_role": role,
        "response_content": "targeted_objection",
        "trigger_kind": trigger,
        "staging": "no_clear_staging_evidence",
        "manual_evidence": "A separate person objects after the action.",
    }


def selection(candidate_id, item_id, uid, cohort="uniform_probability_sample"):
    return {
        "item_id": item_id,
        "uid": uid,
        "cohort": cohort,
        "candidates": [{"candidate_id": candidate_id}],
    }


def model(row, *, error=None):
    return {
        "candidate_id": row["candidate_id"],
        "item_id": row["item_id"],
        "uid": row["uid"],
        "error": error,
        "result": None if error else {
            key: row[key]
            for key in (
                "reaction_grounded",
                "action_before_or_overlaps_response",
                "response_targets_action",
                "responder_role",
                "response_content",
                "trigger_kind",
                "staging",
            )
        },
    }


def prereg(successful=1, expected=1):
    return {"population_model_score_coverage": {
        "expected_candidates": expected,
        "successful_candidates": successful,
        "successful_coverage_fraction": successful / expected,
        "unexpected_candidate_ids": [],
    }}


def test_atomic_values_abstain_on_unresolved_identity_and_uncertain_staging():
    row = manual("a", "clip", "youtube__a", role="offscreen_or_unresolved")
    row["staging"] = "uncertain"
    atoms = atomic_values(row)
    assert atoms["bystander_identity"] is None
    assert atoms["clearly_staged"] is None
    assert atoms["reaction_grounded"] is True


def test_evaluate_reports_uniform_separately_from_enrichment():
    a = manual("a", "clip-a", "youtube__a")
    b = manual(
        "b", "clip-b", "dailymotion__b", grounded="no", role="none",
        trigger="not_established",
    )
    b["response_targets_action"] = "no"
    b["action_before_or_overlaps_response"] = "no"
    b["response_content"] = "none"
    report = evaluate(
        [
            selection("a", "clip-a", "youtube__a"),
            selection(
                "b", "clip-b", "dailymotion__b",
                "predicted_positive_enrichment",
            ),
        ],
        [a, b],
        [model(a), model(b)],
        prereg(2, 2),
    )
    uniform = report["clip_cohort_metrics"]["uniform_probability_sample"]
    enriched = report["clip_cohort_metrics"]["predicted_positive_enrichment"]
    assert uniform["interpretation"] == (
        "unbiased_one_representative_clip_per_source_estimate_not_clip_weighted"
    )
    assert "not_a_corpus_estimate" in enriched["interpretation"]
    assert uniform["routes"]["strict_bystander_reaction"]["fail_closed"]["tp"] == 1
    assert uniform["routes"]["strict_bystander_reaction"]["gold_prevalence"] == 1.0
    assert uniform["routes"]["strict_bystander_reaction"]["gold_prevalence_wilson_95"] is not None
    assert report["automatic_acceptance"] is False
    assert "clip_weighted" not in report["sampling_estimand"]


def test_evaluate_fail_closes_missing_model_candidate():
    row = manual("a", "clip-a", "youtube__a")
    report = evaluate(
        [selection("a", "clip-a", "youtube__a")],
        [row],
        [model(row, error="timeout")],
        prereg(98, 100),
    )
    route = report["clip_cohort_metrics"]["uniform_probability_sample"]["routes"]["strict_bystander_reaction"]
    assert route["fail_closed"]["fn"] == 1
    assert report["selected_sample_model_candidate_coverage"] == 0
    assert report["population_model_candidate_coverage"] == 0.98


def test_evaluate_reroutes_clearly_staged_bystander_like_scene():
    row = manual("a", "clip-a", "youtube__a")
    row["staging"] = "clearly_staged"
    report = evaluate(
        [selection("a", "clip-a", "youtube__a")],
        [row],
        [model(row)],
        prereg(),
    )
    route = report["clip_cohort_metrics"]["uniform_probability_sample"]["routes"]["strict_bystander_social"]
    assert route["fail_closed"]["selected"] == 0
    assert route["fail_closed"]["tn"] == 1
    assert route["manual_staged_instructional_reroute_clips"] == 1


def test_evaluate_rejects_non_source_disjoint_selection():
    a = manual("a", "clip-a", "youtube__same")
    b = manual("b", "clip-b", "youtube__same")
    with pytest.raises(ValueError, match="source-disjoint"):
        evaluate(
            [
                selection("a", "clip-a", "youtube__same"),
                selection("b", "clip-b", "youtube__same"),
            ],
            [a, b],
            [model(a), model(b)],
            prereg(2, 2),
        )


def test_evaluate_rejects_invalid_population_coverage_lineage():
    row = manual("a", "clip-a", "youtube__a")
    bad = prereg(98, 100)
    bad["population_model_score_coverage"]["successful_coverage_fraction"] = 1.0
    with pytest.raises(ValueError, match="population model-coverage"):
        evaluate(
            [selection("a", "clip-a", "youtube__a")], [row], [model(row)], bad
        )


def test_manual_and_model_identity_lineage_must_match_selection():
    row = manual("a", "clip-a", "youtube__a")
    wrong_manual = dict(row, uid="youtube__other")
    with pytest.raises(ValueError, match="manual item_id/uid"):
        evaluate(
            [selection("a", "clip-a", "youtube__a")], [wrong_manual],
            [model(row)], prereg(),
        )
    wrong_model = dict(model(row), item_id="clip-other")
    with pytest.raises(ValueError, match="model item_id/uid"):
        evaluate(
            [selection("a", "clip-a", "youtube__a")], [row],
            [wrong_model], prereg(),
        )


def test_contradictory_manual_reaction_atoms_are_rejected():
    row = manual("a", "clip-a", "youtube__a", grounded="no")
    with pytest.raises(ValueError, match="targeted response must be grounded"):
        evaluate(
            [selection("a", "clip-a", "youtube__a")], [row],
            [model(row)], prereg(),
        )
