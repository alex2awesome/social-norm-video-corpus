import json

from scripts.score_instructional_v11_consensus import parse_result


def _vlm() -> dict:
    return {
        "exact_violation_demo": "yes",
        "performed_violation_not_only_described": "yes",
        "proposed_actor_behavior_target_match": "yes",
        "event_polarity": "violation",
    }


def _consensus() -> dict:
    return {
        "records_agree_same_literal_event": "yes",
        "both_support_performed_violation": "yes",
        "both_support_affected_party": "yes",
        "both_support_exact_norm_actor_target": "yes",
        "both_support_violation_polarity": "yes",
        "hard_exclusion": "none",
        "strict_exact_violation": "yes",
        "failure_mechanism": "none",
        "normalized_literal_event": "child refuses to share with peer",
        "evidence": "Both records describe the same refusal and peer response.",
    }


def test_parse_accepts_strict_consensus() -> None:
    parsed = parse_result(json.dumps(_consensus()), _vlm(), _vlm())
    assert parsed["strict_exact_violation"] == "yes"


def test_parse_requires_both_vlm_inputs_to_be_strict() -> None:
    qwen = _vlm()
    glm = _vlm()
    glm["exact_violation_demo"] = "no"
    parsed = parse_result(json.dumps(_consensus()), qwen, glm)
    assert parsed["strict_exact_violation"] == "uncertain"
    assert "inconsistent_positive" in parsed["validation_repairs"]
