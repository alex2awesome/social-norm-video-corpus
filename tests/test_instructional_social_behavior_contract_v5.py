import copy

import pytest

from scripts.instructional_social_behavior_contract_v5 import validate_result


def social():
    return {
        "social_behavior_domain": "communicative_act",
        "specific_actor_role_grounded": "yes",
        "specific_behavior_grounded": "yes",
        "affected_person_or_shared_social_context_grounded": "yes",
        "normative_obligation_grounded": "yes",
        "literal_social_behavior_not_analogy": "yes",
        "procedural_skill_only": "no",
        "personal_optimization_only": "no",
        "proposed_norm_supported": "yes",
        "proposed_polarity_supported": "yes",
        "normalized_behavior": "one person insults another person's appearance",
        "normalized_norm": "do not insult people for their appearance",
        "behavior_evidence_quote": "you look ugly",
        "norm_evidence_quote": "that is not an acceptable way to speak",
        "route": "strict_social_alignment",
        "description": "Actor, speech act, target, and correction are grounded.",
    }


def test_accepts_literal_social_act_as_text_candidate_not_visual_keep():
    result = validate_result(social())
    assert result["strict_text_candidate"] is True
    assert result["visual_demo_certified"] is False
    assert result["automatic_acceptance"] is False


@pytest.mark.parametrize(
    "domain",
    ["procedural_skill", "personal_optimization", "object_or_product_procedure", "abstract_analogy"],
)
def test_non_social_demonstration_domains_cannot_pass(domain):
    row = social()
    row.update(
        social_behavior_domain=domain,
        route="reject",
        procedural_skill_only="yes" if domain in {"procedural_skill", "object_or_product_procedure"} else "no",
        personal_optimization_only="yes" if domain == "personal_optimization" else "no",
    )
    if domain == "abstract_analogy":
        row["literal_social_behavior_not_analogy"] = "no"
    result = validate_result(row)
    assert result["strict_text_candidate"] is False


def test_grounded_behavior_with_wrong_norm_routes_to_manual_relabel():
    row = social()
    row.update(proposed_norm_supported="no", route="relabel_required")
    result = validate_result(row)
    assert result["text_candidate_after_relabel"] is True
    assert result["strict_text_candidate"] is False


def test_grounded_behavior_with_wrong_or_mixed_polarity_requires_relabel():
    row = social()
    row.update(proposed_polarity_supported="no", route="relabel_required")
    result = validate_result(row)
    assert result["text_candidate_after_relabel"] is True
    assert result["strict_text_candidate"] is False


def test_generic_social_advice_can_only_seed_instructional_retrieval():
    row = social()
    row.update(
        specific_actor_role_grounded="no",
        specific_behavior_grounded="no",
        route="instructional_retrieval_only",
        normalized_behavior="",
        normalized_norm="",
        behavior_evidence_quote="",
        norm_evidence_quote="",
    )
    result = validate_result(row)
    assert result["strict_text_candidate"] is False


def test_route_cannot_override_missing_target_or_analogy_only():
    row = social()
    row["affected_person_or_shared_social_context_grounded"] = "no"
    with pytest.raises(ValueError, match="strict route"):
        validate_result(row)
    row = copy.deepcopy(social())
    row["literal_social_behavior_not_analogy"] = "no"
    with pytest.raises(ValueError, match="strict route"):
        validate_result(row)
