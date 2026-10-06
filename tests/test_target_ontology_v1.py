import pytest

from scripts.target_ontology_v1 import (
    PILLAR_TARGETS,
    SHARED_TARGETS,
    derive_clean_training_span,
    derive_event_audiovisually_grounded,
    derive_independent_bystander_signal,
    derive_label_alignment,
    derive_norm_event_supported,
    derive_reaction_grounded,
    label_disposition,
    pillar_targets,
    tri_and,
)


def test_every_pillar_has_revised_targets():
    assert set(PILLAR_TARGETS) == {"witnessed", "instructional", "commentary"}
    assert "independent_bystander_signal" in PILLAR_TARGETS["witnessed"]
    assert "norm_event_supported" in PILLAR_TARGETS["witnessed"]
    assert "demonstration_present" in PILLAR_TARGETS["instructional"]
    assert "commentary_text_label_supported" in PILLAR_TARGETS["commentary"]
    assert "norm_event_supported" in SHARED_TARGETS


def test_tri_and_is_fail_closed():
    assert tri_and("yes", "yes") == "yes"
    assert tri_and("yes", "uncertain") == "uncertain"
    assert tri_and("uncertain", "no") == "no"
    with pytest.raises(ValueError):
        tri_and("maybe")


def test_situated_speech_counts_as_audiovisual_grounding():
    assert derive_event_audiovisually_grounded("situated_speech_audiovisual") == "yes"
    assert derive_event_audiovisually_grounded("physical_action_visible") == "yes"


def test_description_only_narration_is_not_grounding():
    assert derive_event_audiovisually_grounded("description_only") == "no"
    assert derive_event_audiovisually_grounded("absent") == "no"
    assert derive_event_audiovisually_grounded("uncertain") == "uncertain"
    with pytest.raises(ValueError):
        derive_event_audiovisually_grounded("visible")


def test_norm_event_supported_does_not_require_bystander_identity():
    # An affected-party objection with grounded scope supports the broad target.
    assert (
        derive_norm_event_supported(
            actor_kind="person",
            behavior_kind="physical_action",
            affected_context_kind="person",
            expectation_kind="interpersonal_treatment",
            normative_signal_grounded="yes",
        )
        == "yes"
    )


def test_norm_event_supported_fails_closed_outside_scope():
    assert (
        derive_norm_event_supported(
            actor_kind="nonhuman",
            behavior_kind="physical_action",
            affected_context_kind="person",
            expectation_kind="interpersonal_treatment",
            normative_signal_grounded="yes",
        )
        == "no"
    )
    assert (
        derive_norm_event_supported(
            actor_kind="person",
            behavior_kind="physical_action",
            affected_context_kind="person",
            expectation_kind="interpersonal_treatment",
            normative_signal_grounded="uncertain",
        )
        == "uncertain"
    )


def test_reaction_grounded_requires_temporal_and_causal_links():
    assert (
        derive_reaction_grounded(
            response_observable="yes",
            response_after_or_overlaps="yes",
            response_targets_action="yes",
        )
        == "yes"
    )
    assert (
        derive_reaction_grounded(
            response_observable="yes",
            response_after_or_overlaps="yes",
            response_targets_action="no",
        )
        == "no"
    )


def test_bystander_signal_is_a_subtype_not_a_gate():
    # Affected-party responses fail the subtype but the target still exists
    # separately; the derivation never negates norm_event_supported.
    assert (
        derive_independent_bystander_signal(
            reaction_grounded="yes", responder_role="affected_party"
        )
        == "no"
    )
    assert (
        derive_independent_bystander_signal(
            reaction_grounded="yes", responder_role="independent_bystander"
        )
        == "yes"
    )
    assert (
        derive_independent_bystander_signal(
            reaction_grounded="yes", responder_role="unknown"
        )
        == "uncertain"
    )
    assert (
        derive_independent_bystander_signal(
            reaction_grounded="uncertain", responder_role="organic_audience"
        )
        == "uncertain"
    )
    assert (
        derive_independent_bystander_signal(
            reaction_grounded="no", responder_role="independent_bystander"
        )
        == "no"
    )


def test_repairable_labels_are_preserved_not_rejected():
    assert derive_label_alignment("exact") == "yes"
    assert derive_label_alignment("broad_but_correct") == "yes"
    assert derive_label_alignment("unsupported") == "no"
    assert derive_label_alignment("wrong_but_relabelable") == "uncertain"
    assert label_disposition("wrong_but_relabelable") == "repairable_relabel_review"


def test_clean_training_span_and_pillar_lookup():
    assert (
        derive_clean_training_span(bounds_valid="yes", label_signal_excluded="no")
        == "no"
    )
    assert pillar_targets("witnessed") == PILLAR_TARGETS["witnessed"]
    with pytest.raises(ValueError):
        pillar_targets("negatives")
