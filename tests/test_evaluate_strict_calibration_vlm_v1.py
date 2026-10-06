from scripts.evaluate_strict_calibration_vlm_v1 import (
    commentary_positive,
    instructional_positive,
    witnessed_positive,
)


def test_instructional_requires_every_event_gate() -> None:
    row = {
        "on_screen_social_event": "yes",
        "actor_performs_target_behavior": "yes",
        "affected_party_or_shared_context_same_event": "yes",
        "behavior_socially_evaluable_from_clip": "yes",
        "event_temporally_localized": "yes",
        "informal_social_conduct_not_formal_procedure": "yes",
        "usable_demo_after_relabel": "yes",
        "evidence_source": "physical_action",
        "visual_role": "demonstrated_event",
    }
    assert instructional_positive(row)
    row["event_temporally_localized"] = "uncertain"
    assert not instructional_positive(row)


def test_commentary_requires_clean_localized_scene() -> None:
    row = {
        "social_norm_domain": "yes",
        "situated_social_scenario_visible": "yes",
        "observable_social_behavior_or_speech": "yes",
        "usable_demo_after_relabel": "yes",
        "localization_quality": "clean",
    }
    assert commentary_positive(row)
    row["localization_quality"] = "broad"
    assert not commentary_positive(row)


def test_witnessed_requires_ordered_bounds_and_independent_role() -> None:
    row = {
        "action_visible": "yes",
        "action_voluntary": "yes",
        "expectation_kind": "interpersonal_treatment",
        "reaction_visible_or_audibly_grounded": "yes",
        "reaction_source_role": "bystander",
        "reaction_content": "targeted_objection",
        "action_established_before_reaction": "yes",
        "reaction_targets_action": "yes",
        "authenticity": "organic",
        "pre_reaction_demo_quality": "clear_visual",
        "action_end_percent": 30,
        "reaction_start_percent": 31,
    }
    assert witnessed_positive(row)
    row["reaction_source_role"] = "affected_target"
    assert not witnessed_positive(row)
    row["reaction_source_role"] = "bystander"
    row["reaction_start_percent"] = -1
    assert not witnessed_positive(row)
