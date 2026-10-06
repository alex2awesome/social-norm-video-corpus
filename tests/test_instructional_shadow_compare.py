from pathlib import Path

from scripts.run_instructional_shadow_compare import (
    locate_shadow_demos,
    shadow_contract_rejection_reasons,
    shadow_user_prompt,
)


ROOT = Path(__file__).resolve().parents[1]


def test_shadow_demo_quotes_are_located_and_grounded():
    transcript = {
        "words": [
            {"word": "Alex", "start": 1.0, "end": 1.2},
            {"word": "insulted", "start": 1.2, "end": 1.6},
            {"word": "Sam", "start": 1.6, "end": 1.8},
            {"word": "That", "start": 2.0, "end": 2.2},
            {"word": "was", "start": 2.2, "end": 2.4},
            {"word": "wrong", "start": 2.4, "end": 2.7},
        ]
    }
    result = {
        "demos": [
            {
                "start_quote": "Alex insulted",
                "end_quote": "insulted Sam",
                "label_quote": "That was wrong",
            }
        ]
    }
    demo = locate_shadow_demos(result, transcript)[0]
    assert demo["start_sec"] == 1.0
    assert demo["end_sec"] == 1.8
    assert demo["start_quote_grounded"] is True
    assert demo["end_quote_grounded"] is True
    assert demo["label_quote_grounded"] is True
    assert demo["shadow_contract_pass"] is False
    assert demo["shadow_contract_rejection_reasons"] == [
        "invalid_polarity",
        "invalid_evidence_kind",
        "missing_norm",
        "missing_behavior",
        "missing_actor",
        "missing_target_or_shared_context",
        "insufficient_temporal_context",
    ]


def test_shadow_prompt_marks_title_as_hint_only():
    prompt = shadow_user_prompt("actual transcript", "Misleading demo title")
    assert "retrieval hint only" in prompt
    assert 'Transcript:\n"""actual transcript"""' in prompt


def test_v4_policy_is_medium_neutral_but_demo_strict():
    prompt = (ROOT / "config/instructional_shadow_v4_prompt.txt").read_text()
    normalized = " ".join(prompt.split())
    assert "FORMAT IS NOT A FILTER" in normalized
    assert "exact interval concretely depicts or enacts the labeled social behavior" in normalized
    assert "There is no fixed number of demos per source" in normalized
    assert "training-only labeled-demo tier" in normalized


def test_v4_policy_requires_same_event_polarity_support():
    prompt = (ROOT / "config/instructional_shadow_v4_prompt.txt").read_text()
    assert "Polarity must be independently supported" in prompt
    assert "If narration says an act is appropriate" in prompt
    assert "proposed behavior and label_quote concern different events" in prompt


def valid_demo(**updates):
    demo = {
        "polarity": "violation",
        "evidence_kind": "enacted_dialogue",
        "norm": "do not insult another person",
        "behavior": "Alex insults Sam",
        "actor": "Alex",
        "target_or_shared_context": "Sam",
        "start_quote_grounded": True,
        "end_quote_grounded": True,
        "label_quote_grounded": True,
        "start_sec": 10.0,
        "end_sec": 20.0,
    }
    demo.update(updates)
    return demo


def test_shadow_contract_enforces_reviewed_duration_bounds():
    assert shadow_contract_rejection_reasons(valid_demo()) == []
    assert shadow_contract_rejection_reasons(valid_demo(end_sec=10.8)) == [
        "insufficient_temporal_context"
    ]
    assert shadow_contract_rejection_reasons(valid_demo(end_sec=41.0)) == [
        "duration_exceeds_single_event_limit"
    ]
    assert shadow_contract_rejection_reasons(
        valid_demo(evidence_kind="reported_visible_action", end_sec=51.0)
    ) == []
    assert shadow_contract_rejection_reasons(
        valid_demo(evidence_kind="reported_visible_action", end_sec=56.0)
    ) == ["duration_exceeds_single_event_limit"]


def test_shadow_contract_fails_closed_on_grounding_and_schema():
    reasons = shadow_contract_rejection_reasons(
        valid_demo(polarity="explanation", label_quote_grounded=False, actor="")
    )
    assert reasons == ["invalid_polarity", "missing_actor", "label_quote_ungrounded"]
