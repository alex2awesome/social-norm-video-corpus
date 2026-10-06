from scripts.assess_witnessed_video_asr_corpus_gate import assess


def prereg():
    return {"gate": {
        "minimum_model_candidate_coverage": 0.98,
        "minimum_uniform_clips": 60,
        "minimum_uniform_strict_reaction_precision": 0.70,
        "minimum_uniform_strict_reaction_recall": 0.80,
    }}


def evaluation(precision=0.75, recall=0.85):
    return {
        "source_disjoint": True,
        "manual_candidate_review_complete": True,
        "population_model_candidate_coverage": 1.0,
        "clip_cohort_metrics": {"uniform_probability_sample": {
            "clips": 60,
            "routes": {"strict_bystander_reaction": {"fail_closed": {
                "items": 60,
                "selected": 8,
                "tp": 6,
                "fp": 2,
                "fn": 1,
                "tn": 51,
                "precision": precision,
                "recall": recall,
            }}},
        }},
    }


def test_gate_retains_only_shadow_routing_when_every_check_passes():
    report = assess(prereg(), evaluation())
    assert report["passed"] is True
    assert report["maintenance_action"] == (
        "retain_shadow_candidate_generation_and_review_ranking"
    )
    assert report["automatic_acceptance"] is False


def test_gate_disables_on_precision_failure():
    report = assess(prereg(), evaluation(precision=0.69))
    assert report["passed"] is False
    assert report["maintenance_action"] == "disable_rule"
    assert report["checks"]["uniform_strict_reaction_precision"] is False


def test_gate_fails_closed_on_undefined_recall_or_incomplete_manual_review():
    value = evaluation(recall=None)
    value["manual_candidate_review_complete"] = False
    report = assess(prereg(), value)
    assert report["passed"] is False
    assert report["checks"]["uniform_strict_reaction_recall"] is False
    assert report["checks"]["manual_candidate_review_complete"] is False
