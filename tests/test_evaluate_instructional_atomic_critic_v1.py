from scripts.evaluate_instructional_atomic_critic_v1 import ablation_pass, evaluate
from scripts.run_instructional_atomic_critic_v1 import parse_atomic


def atomic(relation="violates", social_basis="interpersonal_treatment"):
    import json
    text = json.dumps({
        "episode_medium": "roleplay",
        "actor_action": "person insults neighbor",
        "affected_party": "neighbor",
        "grounding": "situated_speech_to_present_party",
        "episode_continuity": "connected",
        "social_basis": social_basis,
        "relation": relation,
        "completeness": "complete",
        "evidence": "The insult is directed at the visible neighbor.",
    })
    return parse_atomic(text, "violation")


def output(item, result):
    return {"item_id": item, "error": None, "result": result}


def test_evaluate_computes_model_consensus_without_promotion() -> None:
    semantics = [
        {"item_id": "a", "uid": "u1", "polarity": "violation"},
        {"item_id": "b", "uid": "u2", "polarity": "violation"},
    ]
    ledger = [
        {"item_id": "a", "visual_form": "roleplay", "usable_demo": "yes"},
        {"item_id": "b", "visual_form": "talking_head", "usable_demo": "no"},
    ]
    good = atomic()
    bad = atomic(social_basis="self_only")
    rows, summary = evaluate(
        semantics,
        ledger,
        {"qwen": [output("a", good), output("b", bad)],
         "gemma": [output("a", good), output("b", good)]},
        2,
    )
    assert summary["rules"]["all_model_consensus"]["precision"] == 1
    assert summary["rules"]["all_model_consensus"]["recall"] == 1
    assert summary["automatic_acceptance"] is False
    assert rows[1]["any_model_union"] is True


def test_ablation_exposes_which_constraint_rejected_item() -> None:
    result = atomic(social_basis="technical_procedure")
    assert not result["atomic_pass"]
    assert ablation_pass(result, "violation", omit="social_basis")
    assert not ablation_pass(result, "violation", omit="relation")
