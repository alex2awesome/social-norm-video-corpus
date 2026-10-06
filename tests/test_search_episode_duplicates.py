import numpy as np

from scripts.audit_search_episode_duplicates import candidate_type, jaccard, pair_metrics, title_tokens


def _record(title, duration, hashes):
    return {
        "title_tokens": title_tokens(title),
        "duration": duration,
        "frame_hashes": [np.asarray(value, dtype=bool) for value in hashes],
    }


def _classify(metrics):
    return candidate_type(
        metrics,
        duration_delta_max=0.005,
        exact_aligned_max=4.0,
        episode_best_max=20.0,
        episode_title_jaccard_min=0.30,
    )


def test_near_exact_reencode_is_flagged():
    hashes = [np.zeros(64, dtype=bool), np.ones(64, dtype=bool)]
    left = _record("same title", 100.0, hashes)
    right = _record("unrelated words", 100.1, hashes)
    assert _classify(pair_metrics(left, right)) == "near_exact_reencode"


def test_lightly_edited_same_episode_requires_title_support():
    base = np.zeros(64, dtype=bool)
    edited = base.copy()
    edited[:18] = True
    left = _record("What Would You Do gender discrimination", 452.54, [base, base])
    right = _record("What Would You Do gender discrimination job interview", 452.54, [edited, edited])
    metrics = pair_metrics(left, right)
    assert jaccard(left["title_tokens"], right["title_tokens"]) >= 0.30
    assert _classify(metrics) == "likely_same_episode_edit"


def test_visual_similarity_without_duration_match_is_not_flagged():
    hashes = [np.zeros(64, dtype=bool), np.ones(64, dtype=bool)]
    left = _record("same title", 100.0, hashes)
    right = _record("same title", 110.0, hashes)
    assert _classify(pair_metrics(left, right)) is None


def test_moderate_visual_similarity_without_title_match_is_not_flagged():
    base = np.zeros(64, dtype=bool)
    edited = base.copy()
    edited[:18] = True
    left = _record("landlord tenant seminar", 100.0, [base, base])
    right = _record("social media conference", 100.0, [edited, edited])
    assert _classify(pair_metrics(left, right)) is None
