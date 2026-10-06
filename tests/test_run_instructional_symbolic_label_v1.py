import json

from scripts.run_instructional_symbolic_label_v1 import (
    SYSTEM_SYMBOLIC, parse_symbolic, semantic_context,
)


def result(**changes):
    value = {
        "observed_behavior": "employee insults customer",
        "claimed_rule": "employees must treat customers respectfully",
        "behavior_match": "yes", "social_basis": "institutional_fairness",
        "relation": "violates", "evidence": "The observed insult breaks the rule.",
    }
    value.update(changes); return json.dumps(value)


def test_symbolic_pass_requires_prior_episode_pass() -> None:
    assert parse_symbolic(result(), True, "violation")["label_pass"] is True
    assert parse_symbolic(result(), False, "violation")["label_pass"] is False


def test_compliant_act_cannot_satisfy_violation_polarity() -> None:
    parsed = parse_symbolic(result(relation="complies"), True, "violation")
    assert parsed["label_pass"] is False


def test_topic_overlap_is_not_behavior_match() -> None:
    parsed = parse_symbolic(result(behavior_match="no", relation="unrelated"), True, "violation")
    assert parsed["label_pass"] is False


def test_technical_and_self_only_rules_fail() -> None:
    for basis in ("technical_procedure", "self_only"):
        assert parse_symbolic(result(social_basis=basis), True, "violation")["label_pass"] is False


def test_context_is_explicit_and_excludes_title() -> None:
    value = semantic_context({
        "norm": "respect", "polarity": "correct", "explanation": "be kind",
        "start_quote": "hello", "end_quote": "thanks", "title": "leading title",
    })
    assert set(value) == {"norm", "polarity", "explanation", "start_quote", "end_quote"}
    assert "only evidence" in SYSTEM_SYMBOLIC
