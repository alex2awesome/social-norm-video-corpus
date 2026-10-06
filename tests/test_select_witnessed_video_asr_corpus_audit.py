from scripts.select_witnessed_video_asr_corpus_audit import (
    model_outcome,
    representative_clips,
    population_score_coverage,
    select,
    successful_scores,
)


def candidate(candidate_id, item_id, uid):
    return {
        "candidate_id": candidate_id,
        "item_id": item_id,
        "uid": uid,
        "candidate_video_path": candidate_id + ".mp4",
        "candidate_video_sha256": "hash",
    }


def score(
    candidate_id, *, role="none", grounded="no", target="no",
    staging="no_clear_staging_evidence", error=None,
):
    return {
        "candidate_id": candidate_id,
        "error": error,
        "result": None if error else {
            "reaction_grounded": grounded,
            "action_before_or_overlaps_response": "yes",
            "response_targets_action": target,
            "responder_role": role,
            "response_content": "targeted_objection",
            "trigger_kind": "interpersonal_treatment",
            "staging": staging,
        },
    }


def test_successful_retry_wins_over_failed_attempt():
    rows = [score("a", error="timeout"), score("a", grounded="yes")]
    assert successful_scores(rows)["a"]["error"] is None


def test_model_outcome_separates_positive_near_miss_and_error():
    rows = [candidate("a", "clip", "youtube__a")]
    assert model_outcome(rows, {}) == "model_error_or_missing"
    near = {"a": score("a", grounded="yes", target="no")}
    assert model_outcome(rows, near) == "predicted_near_miss"
    positive = {
        "a": score(
            "a", role="separate_bystander", grounded="yes", target="yes"
        )
    }
    assert model_outcome(rows, positive) == "predicted_strict_reaction"
    staged = {
        "a": score(
            "a", role="separate_bystander", grounded="yes", target="yes",
            staging="clearly_staged",
        )
    }
    assert model_outcome(rows, staged) == "predicted_near_miss"


def test_population_coverage_uses_successful_retry_not_attempt_count():
    manifest = [candidate("a", "clip-a", "u-a"), candidate("b", "clip-b", "u-b")]
    scores = [score("a", error="timeout"), score("a"), score("b", error="decode")]
    report = population_score_coverage(manifest, scores)
    assert report["expected_candidates"] == 2
    assert report["successful_candidates"] == 1
    assert report["successful_coverage_fraction"] == 0.5


def test_population_coverage_rejects_duplicate_manifest_identity():
    row = candidate("a", "clip-a", "u-a")
    import pytest
    with pytest.raises(ValueError, match="duplicate"):
        population_score_coverage([row, row], [])


def test_representative_population_is_one_clip_per_uid_and_excludes_prior():
    manifest = [
        candidate("a:0", "a:0", "youtube__a"),
        candidate("a:1", "a:1", "youtube__a"),
        candidate("b:0", "b:0", "dailymotion__b"),
    ]
    population = representative_clips(manifest, {"dailymotion__b"}, "seed")
    assert len(population) == 1
    assert population[0]["uid"] == "youtube__a"


def test_select_keeps_uniform_and_enrichment_cohorts_separate_and_disjoint():
    manifest, scores = [], []
    for index in range(12):
        uid = ("youtube" if index % 2 else "dailymotion") + f"__{index}"
        item = f"witnessed:{uid}:0"
        candidate_id = item + ":candidate_0"
        manifest.append(candidate(candidate_id, item, uid))
        if index < 4:
            scores.append(score(
                candidate_id,
                role="separate_bystander",
                grounded="yes",
                target="yes",
            ))
        elif index < 8:
            scores.append(score(candidate_id, grounded="yes", target="no"))
        elif index < 10:
            scores.append(score(candidate_id))
        else:
            scores.append(score(candidate_id, error="decode"))
    sealed, metadata = select(
        manifest,
        scores,
        set(),
        seed="fixed",
        uniform_clips=3,
        positive_enrichment=2,
        near_miss_enrichment=2,
        model_error_enrichment=1,
    )
    assert len({row["uid"] for row in sealed}) == len(sealed)
    assert metadata["selected_clips"] == len(sealed)
    assert metadata["cohorts"]["uniform_probability_sample"] == 3
    assert metadata["eligible_source_uids"] == 12
    assert metadata["eligible_representative_clips"] == 12
    assert metadata["population_model_score_coverage"]["expected_candidates"] == 12
    for row in sealed:
        if row["cohort"] == "predicted_positive_enrichment":
            assert row["model_outcome"] == "predicted_strict_reaction"
        if row["cohort"] == "predicted_near_miss_enrichment":
            assert row["model_outcome"] == "predicted_near_miss"
