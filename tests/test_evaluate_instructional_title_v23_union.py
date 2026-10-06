import pytest

from scripts.evaluate_instructional_title_v23_union import evaluate


def title(item, selected, cohort="fresh"):
    return {"item_id": item, "scene_title_candidate": selected, "cohort": cohort}


def vlm(item, selected, gold):
    return {
        "item_id": item,
        "pipeline_consensus": selected,
        "manual_usable_demo": gold,
    }


def test_union_recovers_complementary_true_positives_without_pooling_labels():
    report = evaluate(
        [title("a", True), title("b", False), title("c", False)],
        [vlm("a", False, True), vlm("b", True, True), vlm("c", False, False)],
        "fresh",
    )
    assert report["rules"]["title_cue"]["recall"] == 0.5
    assert report["rules"]["v23_consensus"]["recall"] == 0.5
    assert report["rules"]["title_or_v23"]["recall"] == 1
    assert report["signal_overlap_selected"] == 0
    assert report["automatic_acceptance"] is False


def test_evaluate_requires_exact_cohort_coverage():
    with pytest.raises(ValueError, match="cohorts differ"):
        evaluate([title("a", True)], [vlm("b", True, True)], "fresh")
