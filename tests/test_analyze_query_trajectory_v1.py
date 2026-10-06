from scripts.analyze_query_trajectory_v1 import normalize_query, summarize, token_jaccard


def query(text, *, active=1, used=1.0, enumerated=0, witnessed=0):
    return {
        "kind": "query",
        "platform": "youtube",
        "query": text,
        "source": "taxonomy",
        "active": active,
        "last_used": used,
        "outcomes": {"enumerated": enumerated, "witnessed": witnessed},
    }


def test_normalization_and_similarity_are_stable():
    assert normalize_query("  Line-cutting: VIDEO! ") == "line cutting video"
    assert token_jaccard("line cutting video", "line cutting incident") == 0.5


def test_summary_detects_saturation_blocked_themes_and_new_proposal():
    rows = [
        query("police confrontation", enumerated=10, witnessed=2),
        query("library noise complaint", enumerated=0),
        query("unused query", used=None),
    ]
    proposed = [{
        "query": "roommate chore conflict role play",
        "pillar": "instructional",
        "canary": True,
    }]
    result = summarize(rows, proposed, [r"\bpolice\b"])
    assert result["used_queries"] == 2
    assert result["never_used_queries"] == 1
    assert result["used_zero_enumerated"] == 1
    assert result["blocked_theme"]["queries"] == 1
    assert result["proposal"]["exact_prior_query_count"] == 0
    assert result["proposal"]["blocked_pattern_match_count"] == 0
