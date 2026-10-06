from scripts.evaluate_instructional_rank_text_gate import evaluate


def test_combination_requires_visual_top_and_strict_text():
    sealed = [
        {"item_id": "strict", "score_band": "top_quintile"},
        {"item_id": "technical", "score_band": "top_quintile"},
        {"item_id": "low", "score_band": "bottom_quintile"},
    ]
    gold = [
        {
            "item_id": "strict",
            "assigned_norm_is_social_norm": "yes",
            "strict_instructional_pass": "yes",
        },
        {
            "item_id": "technical",
            "assigned_norm_is_social_norm": "no",
            "strict_instructional_pass": "no",
        },
        {
            "item_id": "low",
            "assigned_norm_is_social_norm": "yes",
            "strict_instructional_pass": "no",
        },
    ]
    social = {
        "social_norm_candidate": "yes",
        "concrete_behavior_named": "yes",
        "affected_other_or_shared_setting_named": "yes",
        "norm_type": "tacit_interpersonal",
    }
    nonsocial = {
        "social_norm_candidate": "no",
        "concrete_behavior_named": "no",
        "affected_other_or_shared_setting_named": "no",
        "norm_type": "off_topic",
    }
    scores = [
        {"item_id": "strict", "result": social, "error": None},
        {"item_id": "technical", "result": nonsocial, "error": None},
        {"item_id": "low", "result": social, "error": None},
    ]
    report = evaluate(sealed, gold, scores)
    combined = report["combined_top_and_text_strict_vs_strict_pass"]
    assert combined == {
        "tp": 1,
        "fp": 0,
        "fn": 0,
        "tn": 2,
        "selected": 1,
        "precision": 1.0,
        "recall": 1.0,
    }
