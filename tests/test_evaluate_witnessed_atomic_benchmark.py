from scripts.evaluate_witnessed_atomic_benchmark import (
    evaluate,
    instructional_atomic,
    witnessed_atomic,
)


def atomic(**overrides):
    result = {
        "action_visible": "yes",
        "action_voluntary": "yes",
        "expectation_kind": "interpersonal_treatment",
        "reaction_visible_or_audibly_grounded": "yes",
        "reaction_source_role": "bystander",
        "reaction_content": "targeted_objection",
        "action_established_before_reaction": "yes",
        "reaction_targets_action": "yes",
        "authenticity": "organic",
        "proposed_label_relation": "repairable",
        "pre_reaction_demo_quality": "clear_audiovisual",
        "action_end_percent": 40,
        "reaction_start_percent": 50,
    }
    result.update(overrides)
    return result


def test_witnessed_contract_rejects_affected_target_and_generic_affect():
    assert witnessed_atomic(atomic(), require_bounds=True)
    assert not witnessed_atomic(
        atomic(reaction_source_role="affected_target"),
        require_bounds=True,
    )
    assert not witnessed_atomic(
        atomic(reaction_content="generic_affect"),
        require_bounds=True,
    )


def test_exact_band_requires_ordered_nonmissing_bounds():
    assert witnessed_atomic(atomic(), require_bounds=True)
    assert witnessed_atomic(
        atomic(action_end_percent=-1, reaction_start_percent=-1),
        require_bounds=False,
    )
    assert not witnessed_atomic(
        atomic(action_end_percent=-1, reaction_start_percent=-1),
        require_bounds=True,
    )


def test_scripted_scene_routes_only_to_instructional():
    result = atomic(
        authenticity="scripted",
        reaction_source_role="affected_target",
        reaction_content="generic_affect",
        proposed_label_relation="repairable",
    )
    assert not witnessed_atomic(result, require_bounds=False)
    assert instructional_atomic(result)


def test_dual_intersection_is_fail_closed():
    benchmark = [
        {
            "item_id": "a",
            "gold_strict_current": False,
            "gold_witnessed_recovery": True,
            "gold_instructional_reroute": False,
            "gold_any_visual_recovery": True,
        }
    ]
    primary = [{"item_id": "a", "result": atomic()}]
    secondary = [
        {
            "item_id": "a",
            "result": atomic(reaction_source_role="affected_target"),
        }
    ]
    report = evaluate(benchmark, primary, secondary)
    selected = report["rules"]["witnessed_recovery_review_triage"][
        "intersection"
    ]["predicted_positive_items"]
    assert selected == []
