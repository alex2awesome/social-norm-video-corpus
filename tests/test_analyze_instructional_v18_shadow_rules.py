from scripts.analyze_instructional_v18_shadow_rules import (
    RULES,
    feature_record,
)


def record(result: dict) -> dict:
    return {"result": result}


def test_animation_rule_accepts_format_neutral_temporal_demo() -> None:
    source = {
        "item_id": "i",
        "uid": "u",
        "storyboard": {"sampled_timestamps": [0, 10]},
    }
    row = feature_record(
        source,
        record(
            {
                "demo_usable": "yes",
                "scene_role": "animation_or_story_demo",
            }
        ),
        record(
            {
                "scene_role": "animation_or_story_demo",
                "performed_social_behavior": "yes",
                "affected_party_or_shared_setting_present": "yes",
                "socially_evaluable_without_metadata": "yes",
                "social_response_or_consequence_present": "yes",
                "social_scope": "tacit_interpersonal",
            }
        ),
        record(
            {
                "visual_record_supports_concrete_demo": "yes",
                "same_actor_and_target": "yes",
                "quote_role": "performed_dialogue",
            }
        ),
    )
    assert RULES["glm_animation"](row)
    assert RULES["glm_animation_qwen_social"](row)
    assert RULES["glm_animation_qwen_social_response"](row)
    assert RULES["glm_animation_qwen_tacit_or_etiquette"](row)
    assert RULES["animation_or_performed_dialogue"](row)
    assert row["visual_score"] >= 12


def test_presenter_penalty_does_not_become_a_corpus_rejection() -> None:
    source = {
        "item_id": "i",
        "uid": "u",
        "storyboard": {"sampled_timestamps": [0, 120]},
    }
    row = feature_record(
        source,
        record(
            {
                "demo_usable": "yes",
                "scene_role": "situated_scene",
            }
        ),
        record(
            {
                "scene_role": "presenter_sample_or_advice",
                "performed_social_behavior": "no",
                "affected_party_or_shared_setting_present": "no",
            }
        ),
        record(
            {
                "visual_record_supports_concrete_demo": "yes",
                "same_actor_and_target": "no",
                "quote_role": "description_only",
            }
        ),
    )
    assert not RULES["glm_qwen_scene"](row)
    assert row["visual_score"] < 8
