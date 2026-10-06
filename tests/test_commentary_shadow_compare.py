import json

from scripts.run_commentary_shadow_compare import (
    contract_errors,
    evidence_corpus,
    normalized_text,
    user_prompt,
)


ITEM = {
    "quote": "She insulted the ambassador",
    "context": [
        {"text": "She insulted the ambassador, and the host called that unacceptable."}
    ],
}


def accepted(**overrides):
    value = {
        "decision": "accept",
        "same_event": "yes",
        "actor": "she",
        "behavior": "insulted the ambassador",
        "target_or_shared_context": "the ambassador",
        "stance_type": "criticism",
        "proposed_norm_supported": "yes",
        "normalized_norm": "do not personally insult diplomats",
        "behavior_quote": "She insulted the ambassador",
        "stance_quote": "the host called that unacceptable",
        "reject_reason": "",
    }
    value.update(overrides)
    return value


def test_normalized_text_handles_punctuation_and_case():
    assert normalized_text("  That's—Wrong! ") == "that s wrong"


def test_clean_accept_contract_passes():
    assert contract_errors(accepted(), ITEM) == []


def test_accept_requires_same_event():
    assert "accepted_without_same_event" in contract_errors(
        accepted(same_event="uncertain"), ITEM
    )


def test_accept_requires_grounded_quotes():
    assert "ungrounded_stance_quote" in contract_errors(
        accepted(stance_quote="an invented judgment"), ITEM
    )


def test_relabel_requires_deficient_proposed_norm():
    errors = contract_errors(
        accepted(decision="accept_after_relabel", proposed_norm_supported="yes"), ITEM
    )
    assert "relabel_with_supported_norm" in errors


def test_reject_requires_reason():
    value = accepted(decision="reject")
    assert "nonaccept_without_reason" in contract_errors(value, ITEM)


def test_current_exporter_fields_reach_shadow_prompt_and_grounding_contract():
    item = {
        "title": "retrieval title",
        "found_by_query": "reaction to vandalism",
        "category": "comm_reaction_video",
        "norm": "warning",
        "polarity": "consequence",
        "start_quote": "Like, you want to get arrested.",
        "start_sec": 10.0,
        "end_sec": 12.0,
        "transcript_context": [
            {"text": "She's single-handedly destroying the entire store."}
        ],
        "detector_statement": {"quote": "fallback must not replace start_quote"},
    }
    payload = json.loads(user_prompt(item).split("Candidate context:\n", 1)[1].rsplit("\nJSON only.", 1)[0])
    assert payload["detector_quote"] == "Like, you want to get arrested."
    assert payload["query_retrieval_hint_only"] == "reaction to vandalism"
    assert payload["detector_signal_retrieval_hint_only"] == "consequence"
    assert payload["timestamped_transcript_context"] == item["transcript_context"]
    corpus = evidence_corpus(item)
    assert normalized_text("destroying the entire store") in corpus
    assert normalized_text("want to get arrested") in corpus
