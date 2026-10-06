from scripts.score_instructional_v15_utterance_anchor import (
    passes_utterance_anchor,
    quoted_utterances,
    utterance_anchor,
)


def test_physical_action_passes_without_quote() -> None:
    result = {
        "evidence_source": "physical_action",
        "literal_action_or_situated_utterance": "takes a toy",
    }
    assert passes_utterance_anchor(result)
    assert utterance_anchor(result)["reason"] == "non_dialogue_visual_action"


def test_dialogue_requires_specific_quoted_content() -> None:
    result = {
        "evidence_source": "situated_dialogue_or_subtitles",
        "literal_action_or_situated_utterance": "woman talks to man",
        "evidence_during": "woman says 'Leave me alone' while man watches",
    }
    scored = utterance_anchor(result)
    assert scored["passes"] is True
    assert scored["evidence_field"] == "evidence_during"
    assert scored["quoted_utterance"] == "Leave me alone"


def test_generic_dialogue_fails() -> None:
    result = {
        "evidence_source": "situated_dialogue_or_subtitles",
        "literal_action_or_situated_utterance": "speaking to the audience",
        "evidence": "the person's expression changes",
    }
    assert not passes_utterance_anchor(result)


def test_apostrophes_are_not_mistaken_for_quotation_marks() -> None:
    text = "the woman's face changes and the man's hand moves"
    assert quoted_utterances(text) == []


def test_one_word_quote_is_too_weak() -> None:
    assert quoted_utterances("she says 'hello'") == []


def test_curly_double_quotes_are_supported() -> None:
    assert quoted_utterances("text reads “stop doing that”") == [
        "stop doing that"
    ]


def test_single_quoted_utterance_can_contain_contraction() -> None:
    text = "she says 'I don't like old people' to her companion"
    assert quoted_utterances(text) == ["I don't like old people"]
