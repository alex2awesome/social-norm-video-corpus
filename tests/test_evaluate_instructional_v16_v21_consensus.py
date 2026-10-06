from scripts.evaluate_instructional_v16_v21_consensus import evaluate


def _manual(index: int, visual: bool) -> dict:
    return {
        "audit_index": index,
        "candidate_id": f"c{index}",
        "item_id": f"i{index}",
        "manual_visual_demo": visual,
        "manual_relabel_usable": visual,
        "manual_exact_original": visual,
        "visual_form": "scene" if visual else "talking_head",
        "blind_visual_note": "note",
    }


def _output(index: int, positive: bool) -> dict:
    yes = "yes" if positive else "no"
    return {
        "item_id": f"i{index}",
        "error": None,
        "result": {
            "demo_usable": yes,
            "performed_social_behavior": yes,
            "actor_action_target_same_event": yes,
            "affected_party_or_shared_setting_present": yes,
            "socially_evaluable_without_metadata": yes,
            "social_scope": (
                "tacit_interpersonal"
                if positive
                else "non_social_or_ordinary_action"
            ),
            "scene_role": (
                "situated_scene" if positive else "generic_broll_or_montage"
            ),
            "evidence": "event" if positive else "none",
        },
    }


def test_preregistered_consensus_rules_are_scored_without_search() -> None:
    manual = [_manual(i, i < 5) for i in range(60)]
    model_rows = {
        name: [_output(i, i < 5) for i in range(60)]
        for name in ("qwen_v16", "glm_v16", "qwen_v21", "glm_v21")
    }
    rows, summary = evaluate(manual, model_rows)
    assert len(rows) == 60
    assert summary["rules"]["v16_strict_consensus"]["passed"] is True
    assert summary["rules"]["cross_rubric_consensus"]["visual"]["precision"] == 1


def test_consensus_fails_when_support_is_too_small() -> None:
    manual = [_manual(i, i == 0) for i in range(60)]
    model_rows = {
        name: [_output(i, i == 0) for i in range(60)]
        for name in ("qwen_v16", "glm_v16", "qwen_v21", "glm_v21")
    }
    _, summary = evaluate(manual, model_rows)
    assert summary["rules"]["v21_transition_core_consensus"]["passed"] is False
