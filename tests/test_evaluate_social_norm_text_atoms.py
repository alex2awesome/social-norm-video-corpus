from scripts.evaluate_social_norm_text_atoms import social_truth


def test_social_truth_excludes_uncertain_manual_labels():
    rows = {
        "a": {"assigned_norm_is_social_norm": "yes"},
        "b": {"assigned_norm_is_social_norm": "no"},
        "c": {"assigned_norm_is_social_norm": "uncertain"},
    }
    ids, truth = social_truth(rows)
    assert ids == ["a", "b"]
    assert truth == {"a": 1, "b": 0}
