import pytest

from scripts.evaluate_commentary_atomic_microclips import (
    atomic_window_class,
    evaluate_atomic_microclips,
    strict_text_pass,
)


def observer(event: bool = True) -> dict:
    return {
        "item_id": "unused",
        "error": None,
        "result": {
            "visually_observable_event": "yes" if event else "no",
            "actor_visible_description": "a person" if event else "none",
            "action_or_situated_speech_description": (
                "pushes another person" if event else "none"
            ),
            "affected_party_or_shared_setting_description": (
                "another person" if event else "none"
            ),
            "metadata_needed_to_identify_action": "no",
            "presentation_or_context_only": "no" if event else "yes",
            "confidence": 0.9,
        },
    }


def alignment(
    *,
    exact: bool = True,
    usable: bool = True,
    domain: str = "tacit_interpersonal",
    response_only: bool = False,
) -> dict:
    return {
        "item_id": "unused",
        "error": None,
        "result": {
            "both_observers_found_event": "yes",
            "observers_describe_same_event": "yes",
            "literal_social_behavior_visible": (
                "yes"
                if domain in {"tacit_interpersonal", "shared_public"}
                else "no"
            ),
            "social_norm_domain_of_intersection": domain,
            "proposed_label_kind": "concrete_conduct",
            "proposed_label_action_visible": "yes" if exact else "no",
            "response_or_aftermath_only_for_proposed_label": (
                "yes" if response_only else "no"
            ),
            "exact_label_supported": "yes" if exact else "no",
            "usable_after_relabel": "yes" if usable else "no",
            "visual_evidence_independent_of_metadata": "yes",
            "failure_reason": "none" if exact else "action_mismatch",
        },
    }


def text_score(passes: bool) -> dict:
    return {
        "item_id": "parent",
        "error": None,
        "result": {
            "social_norm_candidate": "yes" if passes else "no",
            "concrete_behavior_named": "yes",
            "affected_other_or_shared_setting_named": "yes",
            "norm_type": "tacit_interpersonal",
        },
    }


def manual() -> list[dict]:
    return [
        {
            "item_id": "parent",
            "expected_disposition": "dense_followup_commentary",
            "demo_quality": "clear_visual",
        }
    ]


def test_atomic_window_contract_rejects_response_only_and_political_events():
    assert atomic_window_class(observer(), observer(), alignment(), 0.6) == "exact"
    assert (
        atomic_window_class(
            observer(),
            observer(),
            alignment(exact=False, response_only=True),
            0.6,
        )
        == "relabel"
    )
    assert (
        atomic_window_class(
            observer(),
            observer(),
            alignment(
                exact=False,
                usable=False,
                domain="political_or_formal",
            ),
            0.6,
        )
        == "semantic_conflict"
    )


def test_atomic_exact_rejects_contradictory_failure_reason():
    aligned = alignment()
    aligned["result"]["failure_reason"] = "no_event"
    assert (
        atomic_window_class(observer(), observer(), aligned, 0.6)
        == "relabel"
    )


def test_strict_text_gate_requires_operational_social_norm():
    assert strict_text_pass(text_score(True))
    assert not strict_text_pass(text_score(False))
    record = text_score(True)
    record["result"]["norm_type"] = "abstract_value"
    assert not strict_text_pass(record)


def test_parent_rules_keep_exact_and_relabel_separate():
    windows = [
        {
            "item_id": "window:0",
            "parent_item_id": "parent",
            "window_index": 0,
        },
        {
            "item_id": "window:1",
            "parent_item_id": "parent",
            "window_index": 1,
        },
    ]
    observers_a = []
    observers_b = []
    alignments = []
    for item_id, exact in (("window:0", True), ("window:1", False)):
        first = observer()
        second = observer()
        aligned = alignment(exact=exact)
        first["item_id"] = second["item_id"] = aligned["item_id"] = item_id
        observers_a.append(first)
        observers_b.append(second)
        alignments.append(aligned)
    report = evaluate_atomic_microclips(
        manual(),
        windows,
        observers_a,
        observers_b,
        alignments,
        text_rows=[text_score(True)],
    )
    counts = report["parent_window_counts"]["parent"]
    assert counts["exact_windows"] == 1
    assert counts["relabel_windows"] == 1
    assert (
        report["rules"]["atomic_exact_at_least_1"]["all_target_candidates"][
            "true_positive"
        ]
        == 1
    )
    assert (
        report["rules"]["atomic_exact_at_least_2"]["all_target_candidates"][
            "false_negative"
        ]
        == 1
    )


def test_evaluation_rejects_manual_window_parent_id_mismatch():
    with pytest.raises(ValueError, match="manual/window parent item IDs differ"):
        evaluate_atomic_microclips(
            [
                {
                    "item_id": "wrong-parent",
                    "expected_disposition": "text_only",
                    "demo_quality": "none",
                }
            ],
            [
                {
                    "item_id": "window:0",
                    "parent_item_id": "parent",
                    "window_index": 0,
                }
            ],
            [],
            [],
            [],
        )
