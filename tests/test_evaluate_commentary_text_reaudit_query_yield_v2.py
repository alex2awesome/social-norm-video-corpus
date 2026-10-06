from scripts.evaluate_commentary_text_reaudit_query_yield_v2 import evaluate_yield


SHA = "a" * 64


def source(item_id, query_source, platform="youtube", category="comm"):
    return {
        "item_id": item_id,
        "decision": "accept_after_relabel",
        "query_source": query_source,
        "source_platform": platform,
        "category": category,
    }


def review(item_id, decision="accept", route="strict_visual_search"):
    accepted = decision == "accept"
    return {
        "item_id": item_id,
        "v1_decision": "accept_after_relabel",
        "strict_event_label_decision": decision,
        "occurrence_grounding": "bounded_occurrence" if accepted else "generic_or_hypothetical",
        "behavior_specificity": "specific",
        "stance_quality": "unambiguous_external",
        "visual_search_route": route,
        "corrected_behavior": "person takes property" if accepted else "",
        "corrected_norm": "do not steal" if accepted else "",
        "agrees_with_v1": accepted,
        "manual_rationale": "Complete manual judgment.",
    }


def test_reports_strict_and_preserved_yield_without_authorizing_search_change():
    sources = [source("a", "seed"), source("b", "seed"), source("c", "expand")]
    reviews = [
        review("a"),
        review("b", "reject", "instructional_definition_retrieval"),
        review("c", "reject", "none"),
    ]
    result = evaluate_yield(sources, reviews, SHA, SHA, minimum_action_n=2)
    assert result["strict_event_labels"] == 1
    assert result["preserved_retrieval_leads"] == 2
    assert result["by_query_source"]["seed"]["strict_event_label_rate"] == 0.5
    assert result["by_query_source"]["seed"]["meets_minimum_n_for_search_action"] is True
    assert result["by_query_source"]["seed"]["automatic_search_action"] is None
    assert result["automatic_search_change_authorized"] is False
