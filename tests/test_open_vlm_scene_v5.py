import json

from scripts.run_open_vlm_scene_benchmark import parse_json


def test_v5_schema_requires_event_and_exclusion_fields():
    result = {
        "social_norm_domain": "yes",
        "behavior_occurs_in_scene": "yes",
        "affected_party_or_shared_setting_visible": "yes",
        "situated_interaction_complete": "yes",
        "audience_directed_explanation_only": "no",
        "formal_procedure_trait_or_skill_only": "no",
        "illustrated_action_not_just_text": "na",
        "usable_demo_after_relabel": "yes",
        "proposed_norm_supported": "yes",
        "localization_quality": "clean",
        "depiction_type": "enacted_scene",
        "rejection_reason": "none",
        "evidence": "Two characters enact a refusal and response.",
    }
    assert parse_json(json.dumps(result), "v5") == result
    del result["behavior_occurs_in_scene"]
    try:
        parse_json(json.dumps(result), "v5")
    except ValueError:
        pass
    else:
        raise AssertionError("v5 parser accepted an incomplete result")
