import pytest

from scripts.evaluate_commentary_caption_crop_experiment import evaluate


def item(parent, variant):
    return {
        "candidate_id": f"{parent}--{variant}",
        "parent_candidate_id": parent,
        "variant": variant,
        "proxy_clip": f"/{parent}-{variant}.mp4",
        "proxy_clip_sha256": "a" * 64,
    }


def review(parent, variant, **overrides):
    row = {
        "candidate_id": f"{parent}--{variant}",
        "performed_event_visible": "yes",
        "actor_target_grounded": "yes",
        "start_boundary_clean": "yes",
        "end_boundary_clean": "yes",
        "label_bearing_text_absent": "yes",
        "audio_absent": "yes",
        "medium": "live_action",
        "blind_evidence": "event and both people remain visible without text",
    }
    row.update(overrides)
    return row


def test_reports_variant_and_oracle_free_parent_salvage_metrics():
    manifest = [
        item("a", "mild"), item("a", "center"),
        item("b", "mild"), item("b", "center"),
    ]
    blind = [
        review("a", "mild"),
        review("a", "center", actor_target_grounded="no"),
        review("b", "mild", label_bearing_text_absent="no"),
        review("b", "center"),
    ]
    report, accepted = evaluate(
        manifest, blind, minimum_variant_rate=0.5,
        minimum_parent_salvage_rate=1.0,
    )
    assert report["parents_salvaged_by_any_variant"] == 2
    assert report["parent_salvage_rate"] == 1.0
    assert report["variant_metrics"]["mild"]["usable_rate"] == 0.5
    assert report["variant_metrics"]["center"]["usable_rate"] == 0.5
    assert report["passing_fixed_variants"] == ["center", "mild"]
    assert report["shadow_transform_candidate_promoted"] is True
    assert report["automatic_keep_rule_promoted"] is False
    assert len(accepted) == 2


def test_no_oracle_union_promotion_when_each_fixed_variant_misses_threshold():
    manifest = [
        item("a", "mild"), item("a", "center"),
        item("b", "mild"), item("b", "center"),
    ]
    blind = [
        review("a", "mild"),
        review("a", "center", performed_event_visible="no"),
        review("b", "mild", performed_event_visible="no"),
        review("b", "center"),
    ]
    report, _ = evaluate(manifest, blind, minimum_variant_rate=0.8)
    assert report["parent_salvage_rate"] == 1.0
    assert report["passing_fixed_variants"] == []
    assert report["shadow_transform_candidate_promoted"] is False


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("audio_absent", "", "audio_absent"),
        ("medium", "photo", "medium"),
        ("blind_evidence", "", "blind_evidence"),
    ],
)
def test_incomplete_manual_review_fails_closed(field, value, match):
    manifest = [item("a", "mild")]
    blind = [review("a", "mild", **{field: value})]
    with pytest.raises(ValueError, match=match):
        evaluate(manifest, blind)


def test_manifest_and_review_must_match_exactly():
    with pytest.raises(ValueError, match="does not exactly cover"):
        evaluate([item("a", "mild")], [review("b", "mild")])
