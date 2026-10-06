from scripts.evaluate_instructional_vlm_discovery import v6_strict, v8_strict


def test_v6_strict_requires_more_than_event_yes():
    result = {
        "visually_observable_event": "yes",
        "metadata_needed_to_identify_action": "no",
        "presentation_or_context_only": "no",
        "event_start_percent": 10,
        "event_end_percent": 20,
    }
    assert v6_strict(result)
    result["presentation_or_context_only"] = "yes"
    assert not v6_strict(result)


def test_v8_strict_requires_all_atomic_fields():
    result = {
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
    assert v8_strict(result)
    result["event_temporally_localized"] = "uncertain"
    assert not v8_strict(result)
