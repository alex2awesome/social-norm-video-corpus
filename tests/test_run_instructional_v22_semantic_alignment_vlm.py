from scripts.run_instructional_v22_semantic_alignment_vlm import (
    SYSTEM_SITUATED_ALIGNMENT,
    join_manifests,
    parse_alignment,
    semantic_context,
)


def valid_result() -> str:
    return """{
      "observed_visible_behavior": "a person offers a seat",
      "claimed_behavior": "offer a seat to someone who needs it",
      "behavior_alignment": "yes",
      "polarity_alignment": "yes",
      "demonstration_completeness": "complete",
      "label_grounded_in_pixels": "yes",
      "semantic_pass": "yes",
      "evidence": "The complete offer and response are visible."
    }"""


def test_parse_alignment_accepts_strict_match() -> None:
    assert parse_alignment(valid_result(), "correct")["semantic_pass"] == "yes"


def test_parse_alignment_repairs_inconsistent_positive() -> None:
    text = valid_result().replace('"complete"', '"partial"')
    result = parse_alignment(text, "correct")
    assert result["semantic_pass"] == "uncertain"
    assert result["consistency_repair"] == "inconsistent_positive_to_uncertain"


def test_explanation_allows_not_applicable_polarity() -> None:
    text = valid_result().replace(
        '"polarity_alignment": "yes"',
        '"polarity_alignment": "not_applicable"',
    )
    assert parse_alignment(text, "explanation")["semantic_pass"] == "yes"
    assert parse_alignment(text, "violation")["semantic_pass"] == "uncertain"


def test_context_omits_title_and_query_fields() -> None:
    context = semantic_context(
        {
            "norm": "share",
            "polarity": "correct",
            "explanation": "taking turns",
            "start_quote": "watch",
            "end_quote": "now share",
            "title": "leading title",
            "found_by_query": "leading query",
        }
    )
    assert set(context) == {"norm", "polarity", "explanation", "start_quote", "end_quote"}


def test_join_requires_every_storyboard_item() -> None:
    joined = join_manifests(
        [{"item_id": "i1", "sheet_path": "a.jpg"}],
        [{"item_id": "i1", "norm": "share", "polarity": "correct"}],
    )
    assert joined[0]["_semantic"]["norm"] == "share"


def test_situated_speech_rubric_requires_visible_participants_and_social_scope() -> None:
    assert "speaker interacting with the affected" in SYSTEM_SITUATED_ALIGNMENT
    assert "fail direct-to-camera" in SYSTEM_SITUATED_ALIGNMENT
    assert "solo technical/professional" in SYSTEM_SITUATED_ALIGNMENT
    assert "sarcasm, mockery" in SYSTEM_SITUATED_ALIGNMENT
