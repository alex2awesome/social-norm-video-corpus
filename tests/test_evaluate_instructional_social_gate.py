from scripts.evaluate_instructional_social_gate import evaluate, gold_status


def test_noncore_routes_do_not_count_as_true_positive():
    assert (
        gold_status(
            {
                "decision": "accept_after_relabel",
                "social_norm_domain": "broader_prosocial_animal_care",
            }
        )
        == "noncore_reroute"
    )
    assert (
        gold_status(
            {
                "decision": "reroute_non_social_safety",
                "social_norm_domain": "broader_safety_not_core_social",
            }
        )
        == "noncore_reroute"
    )


def test_metrics_separate_core_reject_and_reroute():
    scores = [
        {"item_id": "a", "derived_band": "v7_exact_candidate"},
        {"item_id": "b", "derived_band": "v7_relabel_candidate"},
        {"item_id": "c", "derived_band": "v7_exact_candidate"},
        {"item_id": "d", "derived_band": "v7_semantic_or_domain_review"},
    ]
    gold = [
        {"item_id": "a", "decision": "accept", "social_norm_domain": "yes"},
        {"item_id": "b", "decision": "reject", "social_norm_domain": "yes"},
        {
            "item_id": "c",
            "decision": "accept_after_relabel",
            "social_norm_domain": "broader_animal_welfare",
        },
        {"item_id": "d", "decision": "accept", "social_norm_domain": "yes"},
    ]

    report = evaluate(scores, gold)

    assert report["candidate_metrics"] == {
        "accepted": 3,
        "core_true_positive": 1,
        "reject_false_positive": 1,
        "noncore_reroute": 1,
        "core_precision": 1 / 3,
        "core_recall": 1 / 2,
    }
