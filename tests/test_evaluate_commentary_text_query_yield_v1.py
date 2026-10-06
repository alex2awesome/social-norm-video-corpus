from scripts.evaluate_commentary_text_query_yield_v1 import evaluate


def row(index: int, decision: str, query_source: str):
    return {
        "item_id": f"commentary:u{index}:0", "uid": f"u{index}",
        "decision": decision, "query_source": query_source,
        "source_platform": "youtube", "category": "c",
        "text_label_manual_reviewed": True,
    }


def test_reports_relabel_yield_without_authorizing_search_change():
    report = evaluate([
        row(0, "accept_after_relabel", "taxonomy"),
        row(1, "reject", "taxonomy"),
        row(2, "accept_after_relabel", "commentary"),
    ], minimum_action_n=3)
    assert report["eligible_for_visual_search_rate"] == 2 / 3
    assert report["raw_detector_phrase_precision"] == 0
    assert report["by_query_source"]["taxonomy"]["meets_minimum_n_for_search_action"] is False
    assert report["automatic_search_change_authorized"] is False
