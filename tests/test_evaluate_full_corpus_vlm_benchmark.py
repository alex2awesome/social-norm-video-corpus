from scripts.evaluate_full_corpus_vlm_benchmark import (
    v5_clean_demo,
    v5_exact_label,
    v7c_strict_witnessed,
)


def clean_v5():
    return {
        "social_norm_domain": "yes",
        "behavior_occurs_in_scene": "yes",
        "affected_party_or_shared_setting_visible": "yes",
        "situated_interaction_complete": "yes",
        "audience_directed_explanation_only": "no",
        "formal_procedure_trait_or_skill_only": "no",
        "usable_demo_after_relabel": "yes",
        "proposed_norm_supported": "yes",
        "localization_quality": "clean",
        "rejection_reason": "none",
    }


def strict_v7c():
    return {
        "action_visible": "yes",
        "action_voluntary": "yes",
        "expectation_kind": "interpersonal_treatment",
        "reaction_visible_or_audibly_grounded": "yes",
        "reaction_source_role": "bystander",
        "reaction_content": "targeted_objection",
        "action_established_before_reaction": "yes",
        "reaction_targets_action": "yes",
        "authenticity": "organic",
        "proposed_label_relation": "exact",
        "pre_reaction_demo_quality": "clear_visual",
        "action_end_percent": 30,
        "reaction_start_percent": 40,
    }


def test_v5_clean_and_exact_are_factored():
    row = clean_v5()
    assert v5_clean_demo(row)
    assert v5_exact_label(row)
    row["proposed_norm_supported"] = "no"
    assert v5_clean_demo(row)
    assert not v5_exact_label(row)


def test_v5_clean_demo_fails_on_presentation():
    row = clean_v5()
    row["audience_directed_explanation_only"] = "yes"
    assert not v5_clean_demo(row)


def test_v7c_requires_ordered_reliable_bounds():
    row = strict_v7c()
    assert v7c_strict_witnessed(row)
    row["action_end_percent"] = -1
    row["reaction_start_percent"] = -1
    assert not v7c_strict_witnessed(row)
