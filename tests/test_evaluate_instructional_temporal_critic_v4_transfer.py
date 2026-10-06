import copy

import pytest

from scripts.evaluate_instructional_temporal_critic_v4_transfer import evaluate


def fixture():
    cohort = [{
        "item_id": "instructional:u:0", "uid": "u", "polarity": "violation",
        "category": "instr_family", "source_platform": "youtube",
        "start_sec": 1.0, "end_sec": 11.0,
    }]
    manual = [{
        "item_id": "instructional:u:0", "uid": "u", "visual_demo": "yes",
        "temporal_state_change": "yes",
        "recipient_response_or_coordinated_trajectory": "yes",
        "segment_start_frame": "2", "segment_end_frame": "20",
        "label_alignment": "exact", "manual_description": "One person pushes another.",
        "manual_note": "", "source_audio_review_status": "reviewed",
        "demo_evidence_modalities": "audiovisual_other",
    }]
    model = [{
        "item_id": "instructional:u:0", "model": "m", "error": None,
        "result": {"demo_candidate": True},
    }]
    review = [{
        "audit_index": "0", "item_id": "instructional:u:0", "uid": "u",
        "model": "m", "model_demo_candidate": "yes",
        "structured_prediction_supported": "yes",
        "rationale_supported": "yes", "input_modality_claim_supported": "yes",
        "predicted_segment_bounds_supported": "yes",
        "unsupported_claim_types": "", "manual_rationale": "Supported.",
    }]
    media = [{"item_id": "instructional:u:0", "audio_present": True}]
    return cohort, manual, model, review, media


def test_reports_required_strata_and_no_activation():
    report = evaluate(*fixture())
    assert report["overall"]["fail_closed"]["tp"] == 1
    assert "polarity=violation" in report["strata"]
    assert "duration=5_to_under_15" in report["strata"]
    assert report["activation_after_this_run"] is False
    assert report["requires_exact_segment_recut_and_label_leakage_reaudit"] is True
    assert report["individual_exact_clips_accepted"] == 0


def test_negative_gold_requires_negative_frame_bounds():
    cohort, manual, model, review, media = fixture()
    manual[0]["visual_demo"] = "no"
    manual[0]["label_alignment"] = "no_visual"
    with pytest.raises(ValueError, match="-1 bounds"):
        evaluate(cohort, manual, model, review, media)


def test_requires_every_output_reviewed():
    cohort, manual, model, review, media = fixture()
    bad = copy.deepcopy(review)
    bad[0]["rationale_supported"] = ""
    with pytest.raises(ValueError, match="rationale_supported"):
        evaluate(cohort, manual, model, bad, media)


def test_rejects_changed_output_review_lineage_or_prediction():
    cohort, manual, model, review, media = fixture()
    bad = copy.deepcopy(review)
    bad[0]["model_demo_candidate"] = "no"
    with pytest.raises(ValueError, match="prediction changed"):
        evaluate(cohort, manual, model, bad, media)
