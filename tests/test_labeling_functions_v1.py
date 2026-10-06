import json
from pathlib import Path

import pytest

from scripts.labeling_functions_v1 import (
    atomic_contract_lf_records,
    eligibility_gate,
    make_lf_record,
    registry_lf_records,
    validate_lf_record,
    write_lf_records,
)
from scripts.weak_signal_registry import load_registry

REGISTRY = load_registry(Path("config/audited_weak_signals_v1.json"))


def record(**overrides):
    base = dict(
        item_id="instructional:youtube__abc:2",
        pillar="instructional",
        target="demonstration_present",
        lf_id="test_lf_v1",
        family="event_action",
        vote=1,
    )
    base.update(overrides)
    return make_lf_record(**base)


def test_valid_record_round_trips():
    row = record()
    assert validate_lf_record(row) is row
    assert row["ontology_version"] == "target_ontology_v1"


def test_abstention_requires_a_reason():
    with pytest.raises(ValueError, match="abstain_reason"):
        record(vote=0)
    assert record(vote=0, abstain_reason="missing evidence")["vote"] == 0
    with pytest.raises(ValueError, match="must not carry"):
        record(vote=1, abstain_reason="spurious")


def test_invalid_vote_target_family_and_confidence_rejected():
    with pytest.raises(ValueError, match="vote"):
        record(vote=2)
    with pytest.raises(ValueError, match="target"):
        record(target="independent_bystander_signal")  # witnessed-only target
    with pytest.raises(ValueError, match="family"):
        record(family="vibes")
    with pytest.raises(ValueError, match="confidence"):
        record(confidence=1.5)
    with pytest.raises(ValueError, match="sha256"):
        record(source_artifact_sha256="zz")


def test_failed_transfer_registry_rules_abstain_by_construction():
    outputs = {
        "instructional_v23_qwen_gemma_consensus": True,
        "instructional_demo_adjudicator_v3": True,
    }
    records = registry_lf_records(
        outputs, REGISTRY, item_id="i1", pillar="instructional"
    )
    assert len(records) == 2
    for row in records:
        assert row["vote"] == 0
        assert row["abstain_reason"] == "failed_transfer_registry_enforced"


def test_rules_without_shadow_vote_contract_never_vote():
    outputs = {
        "audited_scene_queries_v1": True,
        "instructional_scene_title_cues_v1": True,
        "instructional_retro_query_source_priority_v1": "retro_instr_scan",
    }
    records = registry_lf_records(
        outputs, REGISTRY, item_id="i1", pillar="instructional"
    )
    assert len(records) == 3
    for row in records:
        assert row["vote"] == 0
        assert row["abstain_reason"] == "retrieval_or_ranking_use_only_never_a_label"


def test_exclusion_rules_vote_against_the_bystander_subtype_only():
    outputs = {"witnessed_authority_exact_span_v3": True}
    (row,) = registry_lf_records(outputs, REGISTRY, item_id="w1", pillar="witnessed")
    assert row["vote"] == -1
    assert row["target"] == "independent_bystander_signal"
    assert row["family"] == "actor_role_binding"
    assert row["confidence"] == 1.0
    untriggered = registry_lf_records(
        {"witnessed_authority_exact_span_v3": False},
        REGISTRY,
        item_id="w1",
        pillar="witnessed",
    )
    assert untriggered[0]["vote"] == 0
    assert untriggered[0]["abstain_reason"] == "rule_not_triggered"


def test_intervention_scan_casts_audited_positive_shadow_vote():
    (row,) = registry_lf_records(
        {"witnessed_clipwide_intervention_scan_v1": True},
        REGISTRY,
        item_id="w1",
        pillar="witnessed",
    )
    assert row["vote"] == 1
    assert row["target"] == "reaction_grounded"
    assert row["family"] == "reaction"
    assert row["confidence"] == pytest.approx(1 / 3)
    assert row["evidence"]["audited_recall"] == 1.0


def test_non_explanation_polarity_votes_with_null_precision():
    (row,) = registry_lf_records(
        {"instructional_non_explanation_review_priority_v1": "violation"},
        REGISTRY,
        item_id="i1",
        pillar="instructional",
    )
    assert row["vote"] == 1
    assert row["target"] == "demonstration_present"
    assert row["confidence"] is None  # precision unstable across cohorts
    assert row["evidence"]["audited_recall"] == pytest.approx(56 / 60)
    # "explanation" polarity is not a trigger value: abstain, never vote -1.
    (untriggered,) = registry_lf_records(
        {"instructional_non_explanation_review_priority_v1": "explanation"},
        REGISTRY,
        item_id="i1",
        pillar="instructional",
    )
    assert untriggered["vote"] == 0
    assert untriggered["abstain_reason"] == "rule_not_triggered"


def test_atomic_contract_families_become_single_votes():
    row = {
        "social_actor_grounded": "yes",
        "concrete_behavior": "yes",
        "target_or_shared_context_grounded": "yes",
        "is_social_norm": "yes",
        "demo_event_observable": "yes",
        "demo_action_complete": "no",
        "actor_action_target_same_event": "uncertain",
    }
    records = atomic_contract_lf_records(row, item_id="i2", pillar="instructional")
    by_lf = {r["lf_id"]: r for r in records}
    assert by_lf["atomic_instructional_social_scope_v1"]["vote"] == 1
    assert by_lf["atomic_instructional_demonstration_v1"]["vote"] == -1
    same_event = by_lf["atomic_instructional_same_event_grounding_v1"]
    assert same_event["vote"] == 0
    assert same_event["abstain_reason"] == "unresolved_atomic_fields"
    # Missing fields abstain rather than invent certainty.
    alignment = by_lf["atomic_instructional_semantic_alignment_v1"]
    assert alignment["vote"] == 0


def test_gates_fail_closed_and_cover_sanitization():
    gate = eligibility_gate(
        {"media_decodes": "yes", "bounds_valid": "no"}, "witnessed"
    )
    assert gate["failed_gates"] == ["bounds_valid"]
    assert gate["eligible"] is False
    gate = eligibility_gate({"media_decodes": True}, "witnessed")
    assert gate["unknown_gates"] == ["bounds_valid"]
    assert gate["eligible"] is False
    gate = eligibility_gate(
        {
            "media_decodes": True,
            "bounds_valid": True,
            "event_interval_present": True,
            "sanitization_required": True,
        },
        "commentary",
    )
    assert gate["failed_gates"] == ["sanitization_verified"]


def test_write_lf_records_is_append_only(tmp_path):
    out = tmp_path / "lf.jsonl"
    assert write_lf_records([record()], out) == 1
    with pytest.raises(FileExistsError):
        write_lf_records([record()], out)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert rows[0]["lf_id"] == "test_lf_v1"
