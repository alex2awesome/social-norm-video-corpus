from scripts.evaluate_visual_ranker_confirmation import merge_reviews, truth_for


def test_dense_review_overrides_blind_uncertainty():
    blind = [{"item_id": "x", "visual_demo_present": "uncertain"}]
    dense = [{"item_id": "x", "visual_demo_present": "yes"}]
    assert merge_reviews(blind, dense)["x"]["visual_demo_present"] == "yes"


def test_witnessed_truth_keeps_organic_separate_from_scene():
    truth = truth_for(
        "witnessed",
        {
            "human_event_visible": "yes",
            "social_interaction_scene_visible": "yes",
            "witnessed_candidate_blind": "no",
        },
    )
    assert truth == {
        "human_event": 1,
        "social_interaction_scene": 1,
        "organic_witnessed_candidate": 0,
    }
