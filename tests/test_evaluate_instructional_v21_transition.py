from scripts.evaluate_instructional_v21_transition import evaluate


def _manual(index: int, visual: bool, usable: bool) -> dict:
    return {
        "audit_index": index,
        "candidate_id": f"c{index}",
        "item_id": f"i{index}",
        "manual_visual_demo": visual,
        "manual_relabel_usable": usable,
        "manual_exact_original": visual and usable,
        "visual_form": "scene" if visual else "talking_head",
        "blind_visual_note": "manual",
    }


def _vlm(index: int, accept: bool) -> dict:
    return {
        "item_id": f"i{index}",
        "model": "qwen",
        "rubric": "v21_event_transition",
        "error": None,
        "result": {
            "demo_usable": "yes" if accept else "no",
            "scene_role": "situated_scene",
            "social_scope": "tacit_interpersonal",
            "rejection_reason": "none" if accept else "behavior_only_described",
            "literal_action_or_situated_utterance": "insults target",
            "evidence": "earlier setup, then insult, later response",
        },
    }


def test_v21_evaluation_fails_below_precision_gate() -> None:
    manual = [_manual(i, i < 5, i < 5) for i in range(60)]
    vlm = [_vlm(i, i < 6) for i in range(60)]
    rows, summary = evaluate(manual, vlm, minimum_precision=0.9)
    assert len(rows) == 60
    assert summary["metrics"]["visual"]["precision"] == 5 / 6
    assert summary["gate"]["passed"] is False
    assert summary["visual_false_positives"][0]["audit_index"] == 5


def test_v21_evaluation_passes_only_with_support_and_two_precision_gates() -> None:
    manual = [_manual(i, i < 5, i < 5) for i in range(60)]
    vlm = [_vlm(i, i < 5) for i in range(60)]
    _, summary = evaluate(manual, vlm, minimum_precision=0.9)
    assert summary["gate"]["passed"] is True
    _, low_support = evaluate(manual, vlm, minimum_support=6)
    assert low_support["gate"]["passed"] is False
