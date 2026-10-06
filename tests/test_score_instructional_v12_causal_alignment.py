import json

from scripts.score_instructional_v12_causal_alignment import parse_result


def _record() -> dict:
    return {
        "visual_record_supports_concrete_demo": "yes",
        "proposed_norm_names_literal_action": "yes",
        "same_actor_and_target": "yes",
        "causal_or_intent_match": "not_required",
        "proposed_violation_polarity_matches": "yes",
        "quote_role": "performed_dialogue",
        "norm_relation": "exact",
        "strict_exact_alignment": "yes",
        "failure_mechanism": "none",
        "normalized_visible_action": "child interrupts teacher",
        "evidence": "The visible interruption matches the proposed norm.",
    }


def test_parse_accepts_consistent_positive() -> None:
    result = parse_result(json.dumps(_record()))
    assert result["strict_exact_alignment"] == "yes"


def test_parse_repairs_inconsistent_positive() -> None:
    record = _record()
    record["causal_or_intent_match"] = "no"
    result = parse_result(json.dumps(record))
    assert result["strict_exact_alignment"] == "uncertain"
    assert result["validation_repair"] == "inconsistent_positive"
