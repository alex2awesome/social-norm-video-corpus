import pytest

from scripts.evaluate_commentary_ocr_line_mask_experiment import evaluate_with_abstentions
from tests.test_evaluate_commentary_caption_crop_experiment import item, review


def test_abstentions_are_included_in_cohort_rate_and_block_promotion():
    manifest = [item("a", "ocr_line_mask_v2")]
    blind = [review("a", "ocr_line_mask_v2")]
    summary = {
        "cohort_items": 4,
        "renderable_items": 1,
        "abstentions": [
            {"parent_candidate_id": "b", "reason": "unsafe"},
            {"parent_candidate_id": "c", "reason": "unsafe"},
            {"parent_candidate_id": "d", "reason": "unsafe"},
        ],
    }
    report, accepted = evaluate_with_abstentions(manifest, blind, summary)
    assert len(accepted) == 1
    assert report["cohort_usable_rate"] == 0.25
    assert report["shadow_transform_candidate_promoted"] is False
    assert "artifact_sha256" not in report


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("renderable_items", 2, "renderable count"),
        ("cohort_items", 5, "cohort does not equal"),
    ],
)
def test_plan_summary_accounting_fails_closed(field, value, match):
    manifest = [item("a", "ocr_line_mask_v2")]
    blind = [review("a", "ocr_line_mask_v2")]
    summary = {
        "cohort_items": 2,
        "renderable_items": 1,
        "abstentions": [{"parent_candidate_id": "b", "reason": "unsafe"}],
    }
    summary[field] = value
    with pytest.raises(ValueError, match=match):
        evaluate_with_abstentions(manifest, blind, summary)
