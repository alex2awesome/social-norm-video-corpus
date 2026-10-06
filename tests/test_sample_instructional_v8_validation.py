from scripts.sample_instructional_v8_validation import balanced_source_disjoint


def test_balanced_selection_is_uid_disjoint():
    rows = [
        {
            "item_id": f"item:{index}",
            "uid": f"uid:{index // 2}",
            "category": f"cat:{index % 2}",
            "polarity": "correct",
            "v8_vote_pattern": "both_positive",
        }
        for index in range(8)
    ]
    selected = balanced_source_disjoint(
        rows,
        4,
        7,
        include_pattern=False,
        used_uids=set(),
    )
    assert len(selected) == 4
    assert len({row["uid"] for row in selected}) == 4


def test_balanced_rejects_include_vote_pattern_in_strata():
    rows = [
        {
            "item_id": f"item:{pattern}",
            "uid": f"uid:{pattern}",
            "category": "cat",
            "polarity": "correct",
            "v8_vote_pattern": pattern,
        }
        for pattern in ("qwen_only", "glm_only", "both_reject")
    ]
    selected = balanced_source_disjoint(
        rows,
        3,
        7,
        include_pattern=True,
        used_uids=set(),
    )
    assert {row["v8_vote_pattern"] for row in selected} == {
        "qwen_only",
        "glm_only",
        "both_reject",
    }
