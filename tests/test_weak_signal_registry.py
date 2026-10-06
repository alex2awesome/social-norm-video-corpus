import json
from pathlib import Path

import pytest

from scripts.weak_signal_registry import combine_signals, load_registry, validate_registry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "config" / "audited_weak_signals_v1.json"


def rule_by_id(registry: dict, rule_id: str) -> dict:
    return next(rule for rule in registry["rules"] if rule["rule_id"] == rule_id)


def metric_tuple(value: dict) -> tuple:
    return (
        value["tp"],
        value["fp"],
        value["fn"],
        value.get("selected", value.get("predicted_positive")),
        value["precision"],
        value["recall"],
    )


def test_real_registry_has_frozen_evidence_and_no_acceptance_gate() -> None:
    registry = load_registry(REGISTRY_PATH)
    summary = validate_registry(registry, ROOT)
    assert summary == {
        "schema_version": 1,
        "rules": 28,
        "active_rules": 10,
        "failed_transfer_rules": 18,
        "acceptance_gate_rules": 0,
        "automatic_keep_enabled": False,
        "policy": "append_only_non_destructive_weak_supervision",
    }


def test_combiner_routes_and_excludes_without_keep_or_delete() -> None:
    registry = load_registry(REGISTRY_PATH)
    result = combine_signals(
        {
            "witnessed_clipwide_intervention_scan_v1": True,
            "witnessed_authority_exact_span_v3": True,
            "witnessed_creator_staging_title_v2": False,
            "instructional_v20_multimodal_rank_high": "high",
            "unknown_experiment": True,
        },
        registry,
    )
    assert result["triggered_rule_ids"] == [
        "witnessed_authority_exact_span_v3",
        "witnessed_clipwide_intervention_scan_v1",
    ]
    assert result["review_routes"] == [
        "reaction_localization_review",
        "visible_scene_review",
    ]
    assert result["strict_route_exclusions"] == ["strict_organic_witnessed"]
    assert result["acceptance_label"] is None
    assert result["corpus_disposition"] is None
    assert result["delete_media"] is False
    assert result["unknown_signal_ids"] == ["unknown_experiment"]


def test_retro_instructional_provenance_only_adds_manual_review_priority() -> None:
    registry = load_registry(REGISTRY_PATH)
    result = combine_signals(
        {"instructional_retro_query_source_priority_v1": "retro_instr_scan"},
        registry,
    )
    assert result["triggered_rule_ids"] == [
        "instructional_retro_query_source_priority_v1"
    ]
    assert result["review_routes"] == ["instructional_demo_manual_review"]
    assert result["rankings"] == [{
        "rule_id": "instructional_retro_query_source_priority_v1",
        "priority_tier": "high_review",
    }]
    assert result["acceptance_label"] is None
    assert result["corpus_disposition"] is None
    assert result["delete_media"] is False


def test_non_explanation_only_changes_instructional_review_order() -> None:
    registry = load_registry(REGISTRY_PATH)
    ranked = combine_signals(
        {"instructional_non_explanation_review_priority_v1": "violation"},
        registry,
    )
    assert ranked["triggered_rule_ids"] == [
        "instructional_non_explanation_review_priority_v1"
    ]
    assert ranked["review_routes"] == ["instructional_demo_manual_review"]
    assert ranked["rankings"] == [{
        "rule_id": "instructional_non_explanation_review_priority_v1",
        "priority_tier": "standard_above_explanation",
    }]
    assert ranked["strict_route_exclusions"] == []
    assert ranked["acceptance_label"] is None
    assert ranked["corpus_disposition"] is None
    assert ranked["delete_media"] is False

    explanation = combine_signals(
        {"instructional_non_explanation_review_priority_v1": "explanation"},
        registry,
    )
    assert explanation["triggered_rule_ids"] == []
    assert explanation["corpus_disposition"] is None


def test_failed_transfer_rule_cannot_trigger() -> None:
    registry = load_registry(REGISTRY_PATH)
    result = combine_signals(
        {"witnessed_identity_multimodal_ensemble_v4": True}, registry
    )
    assert result["triggered_rule_ids"] == []
    assert result["review_routes"] == []

    stale_qwen = combine_signals(
        {"witnessed_qwen_video_asr_reaction_retrieval_v3": True}, registry
    )
    assert stale_qwen["triggered_rule_ids"] == []
    assert stale_qwen["review_routes"] == []

    instructional = combine_signals(
        {"instructional_v20_multimodal_rank_high": "high"}, registry
    )
    assert instructional["triggered_rule_ids"] == []
    assert instructional["review_routes"] == []

    union = combine_signals(
        {"instructional_title_or_v23_union_v1": True}, registry
    )
    assert union["triggered_rule_ids"] == []
    assert union["review_routes"] == []

    blind_episode = combine_signals(
        {"instructional_qwen_blind_episode_relaxed_v1": True}, registry
    )
    assert blind_episode["triggered_rule_ids"] == []
    assert blind_episode["review_routes"] == []

    demo_consensus = combine_signals(
        {"instructional_qwen_demo_consensus_v2": True}, registry
    )
    assert demo_consensus["triggered_rule_ids"] == []
    assert demo_consensus["review_routes"] == []

    metadata_cues = combine_signals(
        {"instructional_metadata_script_cues_v1": True}, registry
    )
    assert metadata_cues["triggered_rule_ids"] == []
    assert metadata_cues["review_routes"] == []

    v23 = combine_signals(
        {"instructional_v23_qwen_gemma_consensus": True}, registry
    )
    assert v23["triggered_rule_ids"] == []
    assert v23["review_routes"] == []

    failed_commentary = combine_signals(
        {
            "commentary_fixed_caption_crop_v1": True,
            "commentary_motion_bbox_crop_v1": True,
            "commentary_multilingual_ocr_mask_v1": True,
            "commentary_multilingual_ocr_line_mask_v2": True,
            "commentary_fixed_tail_trim_v1": True,
        },
        registry,
    )
    assert failed_commentary["triggered_rule_ids"] == []
    assert failed_commentary["review_routes"] == []


def test_official_wwyd_rule_only_excludes_organic_and_routes_for_review() -> None:
    registry = load_registry(REGISTRY_PATH)
    result = combine_signals(
        {"witnessed_official_wwyd_channel_v1": True}, registry
    )
    assert result["triggered_rule_ids"] == [
        "witnessed_official_wwyd_channel_v1"
    ]
    assert result["review_routes"] == ["instructional_demo_review"]
    assert result["strict_route_exclusions"] == ["strict_organic_witnessed"]
    assert result["acceptance_label"] is None
    assert result["corpus_disposition"] is None
    assert result["delete_media"] is False


def test_active_rule_cannot_use_non_disjoint_evidence() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    rule = rule_by_id(registry, "commentary_fixed_tail_trim_v1")
    rule["status"] = "audited_for_declared_use"
    rule["allowed_uses"] = ["candidate_generation"]
    rule["trigger_values"] = [True]
    with pytest.raises(ValueError, match="not source-disjoint"):
        validate_registry(registry, ROOT)


def test_acceptance_use_fails_without_strong_promotion_contract() -> None:
    registry = load_registry(REGISTRY_PATH)
    rule = registry["rules"][0]
    rule["allowed_uses"] = ["acceptance_gate"]
    registry["acceptance_gate_rules"] = 1
    with pytest.raises(ValueError, match="acceptance promotion contract failed"):
        validate_registry(registry)


def test_metric_inconsistency_is_rejected() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    evidence = registry["rules"][1]["evidence"][0]
    evidence["precision"] = 0.99
    with pytest.raises(ValueError, match="inconsistent precision"):
        validate_registry(registry)


def test_evidence_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    evidence = registry["rules"][0]["evidence"][0]
    artifact = tmp_path / evidence["artifact"]
    artifact.parent.mkdir(parents=True)
    artifact.write_text("changed\n")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_registry(registry, tmp_path)


def test_registered_metrics_match_frozen_evaluations() -> None:
    registry = load_registry(REGISTRY_PATH)

    candidate = rule_by_id(
        registry, "witnessed_clipwide_intervention_scan_v1"
    )["evidence"][0]
    candidate_report = json.loads((ROOT / candidate["artifact"]).read_text())
    candidate_gold = candidate_report["frozen_rules"]["strict_bystander"][
        "metrics"
    ]["clip_scan_candidate_generator"]
    assert metric_tuple(candidate) == (
        candidate_gold["true_positive"],
        candidate_gold["false_positive"],
        candidate_gold["false_negative"],
        candidate_gold["predicted_positive"],
        candidate_gold["precision"],
        candidate_gold["recall"],
    )

    av_reaction_evidence = rule_by_id(
        registry, "witnessed_qwen_video_asr_reaction_retrieval_v3"
    )["evidence"]
    av_reaction = av_reaction_evidence[0]
    av_report = json.loads((ROOT / av_reaction["artifact"]).read_text())
    av_gold = av_report["metrics"]
    assert metric_tuple(av_reaction) == (
        av_gold["tp"],
        av_gold["fp"],
        av_gold["fn"],
        av_gold["selected"],
        av_gold["precision"],
        av_gold["recall"],
    )
    av_uniform = av_reaction_evidence[1]
    av_uniform_report = json.loads((ROOT / av_uniform["artifact"]).read_text())
    av_uniform_gold = av_uniform_report["clip_cohort_metrics"][
        "uniform_probability_sample"
    ]["routes"]["strict_bystander_reaction"]["fail_closed"]
    assert metric_tuple(av_uniform) == metric_tuple(av_uniform_gold)
    av_reaction_only = av_reaction_evidence[2]
    av_reaction_only_report = json.loads(
        (ROOT / av_reaction_only["artifact"]).read_text()
    )
    av_reaction_only_gold = av_reaction_only_report["clip_cohort_metrics"][
        "uniform_probability_sample"
    ]["routes"]["strict_bystander_reaction_staging_independent"]["fail_closed"]
    assert metric_tuple(av_reaction_only) == metric_tuple(av_reaction_only_gold)
    assert av_reaction_only["staging_independent_reaction_target"] is True

    authority = rule_by_id(
        registry, "witnessed_authority_exact_span_v3"
    )["evidence"]
    for evidence, model_name in zip(authority, ("v3_exact_span", "v3")):
        report = json.loads((ROOT / evidence["artifact"]).read_text())
        assert metric_tuple(evidence) == metric_tuple(report["models"][model_name])

    staging = rule_by_id(
        registry, "witnessed_creator_staging_title_v2"
    )["evidence"]
    first = json.loads((ROOT / staging[0]["artifact"]).read_text())["rules"][
        "title_prank_social_experiment_hidden_camera"
    ]
    second = json.loads((ROOT / staging[1]["artifact"]).read_text())[
        "refined_rule"
    ]
    assert metric_tuple(staging[0]) == metric_tuple(first)
    assert metric_tuple(staging[1]) == metric_tuple(second)

    wwyd = rule_by_id(
        registry, "witnessed_official_wwyd_channel_v1"
    )["evidence"][0]
    wwyd_report = json.loads((ROOT / wwyd["artifact"]).read_text())
    wwyd_gold = wwyd_report["rules"]["exact_official_wwyd_channel"]
    assert metric_tuple(wwyd) == (
        wwyd_gold["true_positive"],
        wwyd_gold["false_positive"],
        wwyd_report["gold_source_nonorganic_produced"]
        - wwyd_gold["true_positive"],
        wwyd_gold["selected_sources"],
        wwyd_gold["precision"],
        wwyd_gold["true_positive"]
        / wwyd_report["gold_source_nonorganic_produced"],
    )

    commentary_evidence = rule_by_id(
        registry, "commentary_dual_vlm_retrieval_core_v1"
    )["evidence"]
    commentary = commentary_evidence[0]
    commentary_report = json.loads((ROOT / commentary["artifact"]).read_text())
    commentary_gold = commentary_report["slices"]["all_154"]["rules"][
        "dual_retrieval_core"
    ]["usable_any_route"]
    assert metric_tuple(commentary) == metric_tuple(commentary_gold)

    commentary_transfer = commentary_evidence[1]
    transfer_report = json.loads(
        (ROOT / commentary_transfer["artifact"]).read_text()
    )
    assert transfer_report["review_ranking_rule_retained"] is True
    assert transfer_report["manual_model_outputs_reviewed"] == 46
    assert metric_tuple(commentary_transfer) == metric_tuple(
        transfer_report["metrics"]
    )

    polarity = rule_by_id(
        registry, "instructional_non_explanation_review_priority_v1"
    )["evidence"]
    polarity_report = json.loads((ROOT / polarity[0]["artifact"]).read_text())
    assert metric_tuple(polarity[0]) == metric_tuple(
        polarity_report["cohorts"]["fresh_100"]
    )
    assert metric_tuple(polarity[1]) == metric_tuple(
        polarity_report["cohorts"]["replication_100"]
    )


def test_registered_ranking_denominators_match_manual_audits() -> None:
    registry = load_registry(REGISTRY_PATH)
    search = rule_by_id(registry, "audited_scene_queries_v1")["evidence"][0]
    search_rows = [
        line
        for line in (ROOT / search["artifact"]).read_text().splitlines()
        if line.strip()
    ]
    assert len(search_rows) == search["reviewed"] == 60

    instructional = rule_by_id(
        registry, "instructional_v20_multimodal_rank_high"
    )["evidence"][0]
    instructional_report = json.loads(
        (ROOT / instructional["artifact"]).read_text()
    )
    assert instructional_report["overall"]["items"] == instructional["reviewed"]
    assert instructional_report["bands"]["high"]["visual_demos"] == 10
    assert instructional_report["bands"]["high"]["items"] == 30
    assert instructional_report["bands"]["low"]["visual_demos"] == 0
    assert instructional_report["bands"]["low"]["items"] == 15

    title = rule_by_id(registry, "commentary_capture_title_event_v1")[
        "evidence"
    ][0]
    title_report = json.loads((ROOT / title["artifact"]).read_text())
    assert title_report["population"] == title["reviewed"] == 154
    assert title_report["source_usable_candidates"] == 93
    assert title_report["strict_exact_title_event_candidates"] == 74


def test_failed_commentary_remediation_denominators_match_manual_audits() -> None:
    registry = load_registry(REGISTRY_PATH)
    expected = {
        "commentary_fixed_caption_crop_v1": (22, 11),
        "commentary_motion_bbox_crop_v1": (11, 11),
        "commentary_multilingual_ocr_mask_v1": (11, 11),
    }
    for rule_id, (reviewed, parents) in expected.items():
        evidence = rule_by_id(registry, rule_id)["evidence"][0]
        report = json.loads((ROOT / evidence["artifact"]).read_text())
        assert evidence["reviewed"] == report["items"] == reviewed
        assert evidence["parents"] == report["parents"] == parents
        assert report["shadow_transform_candidate_promoted"] is False

    guarded = rule_by_id(
        registry, "commentary_multilingual_ocr_line_mask_v2"
    )["evidence"][0]
    guarded_report = json.loads((ROOT / guarded["artifact"]).read_text())
    assert guarded["reviewed"] == guarded_report["cohort_items"] == 11
    assert guarded["manual_rendered_outputs_reviewed"] == guarded_report["items"] == 3
    assert guarded["safety_abstentions"] == guarded_report["abstention_count"] == 8
    assert guarded_report["shadow_transform_candidate_promoted"] is False


def test_failed_commentary_exact_clip_rule_matches_complete_manual_audit() -> None:
    registry = load_registry(REGISTRY_PATH)
    evidence = rule_by_id(
        registry, "commentary_dual_vlm_exact_clip_v1"
    )["evidence"][0]
    report = json.loads((ROOT / evidence["artifact"]).read_text())
    assert report["manual_review_complete"] is True
    assert report["automatic_acceptance_rule_promoted"] is False
    assert evidence["reviewed"] == report["items"] == 32
    assert (
        evidence["exact_commentary_visual_clips"]
        == report["exact_commentary_visual_clips"]
        == 2
    )


def test_failed_adaptive_edge_crop_includes_abstentions() -> None:
    registry = load_registry(REGISTRY_PATH)
    evidence = rule_by_id(
        registry, "commentary_adaptive_edge_crop_v1"
    )["evidence"][0]
    report = json.loads((ROOT / evidence["artifact"]).read_text())
    assert evidence["reviewed"] == report["cohort_items"] == 11
    assert (
        evidence["manual_rendered_outputs_reviewed"]
        == report["manual_outputs_reviewed"]
        == 2
    )
    assert evidence["safety_abstentions"] == report["abstention_count"] == 9
    assert report["candidate_for_fresh_source_disjoint_holdout"] is False


def test_failed_instructional_visual_title_conjunction_matches_intensive_audit() -> None:
    registry = load_registry(REGISTRY_PATH)
    rule = rule_by_id(registry, "instructional_visual_title_conjunction_v1")
    evidence = rule["evidence"][0]
    report = json.loads((ROOT / evidence["artifact"]).read_text())
    metric = report["metrics"]["visual_title_conjunction"]
    assert report["manual_review_complete"] is True
    assert report["manual_model_outputs_reviewed"] == 196
    assert report["preregistered_pass"] is False
    assert rule["status"] == "failed_transfer"
    assert rule["allowed_uses"] == []
    assert evidence["reviewed"] == report["items"] == 100
    assert evidence["rendered"] == report["rendered_items"] == 98
    assert metric_tuple(evidence) == metric_tuple(metric)
    assert report["automatic_acceptance"] is False
    assert report["corpus_mutation_authorized"] is False


def test_shadow_lf_vote_contracts_are_validated() -> None:
    registry = load_registry(REGISTRY_PATH)
    voting = [
        rule
        for rule in registry["rules"]
        if "shadow_lf_vote" in (rule.get("allowed_uses") or [])
    ]
    assert {rule["rule_id"] for rule in voting} == {
        "witnessed_clipwide_intervention_scan_v1",
        "witnessed_authority_exact_span_v3",
        "witnessed_creator_staging_title_v2",
        "witnessed_official_wwyd_channel_v1",
        "instructional_non_explanation_review_priority_v1",
        "commentary_capture_title_event_v1",
        "commentary_dual_vlm_retrieval_core_v1",
    }
    for rule in voting:
        contract = rule["shadow_lf_vote_contract"]
        assert contract["shadow_only"] is True
        assert contract["vote_when_triggered"] in (-1, 1)
        assert rule["status"] == "audited_for_declared_use"
    # Exclusion cues vote only against the strict bystander subtype.
    for rule_id in (
        "witnessed_authority_exact_span_v3",
        "witnessed_creator_staging_title_v2",
        "witnessed_official_wwyd_channel_v1",
    ):
        contract = rule_by_id(registry, rule_id)["shadow_lf_vote_contract"]
        assert contract["target"] == "independent_bystander_signal"
        assert contract["vote_when_triggered"] == -1


def test_shadow_lf_vote_requires_complete_contract() -> None:
    registry = load_registry(REGISTRY_PATH)
    rule = rule_by_id(registry, "witnessed_clipwide_intervention_scan_v1")
    broken = json.loads(json.dumps(registry))
    del rule_by_id(broken, rule["rule_id"])["shadow_lf_vote_contract"]
    with pytest.raises(ValueError, match="contract"):
        validate_registry(broken)
    broken = json.loads(json.dumps(registry))
    rule_by_id(broken, rule["rule_id"])["shadow_lf_vote_contract"]["vote_when_triggered"] = 0
    with pytest.raises(ValueError, match="vote_when_triggered"):
        validate_registry(broken)
    broken = json.loads(json.dumps(registry))
    rule_by_id(broken, rule["rule_id"])["shadow_lf_vote_contract"]["shadow_only"] = False
    with pytest.raises(ValueError, match="shadow_only"):
        validate_registry(broken)
    broken = json.loads(json.dumps(registry))
    rule_by_id(broken, rule["rule_id"])["shadow_lf_vote_contract"]["audited_precision"] = 1.5
    with pytest.raises(ValueError, match="audited_precision"):
        validate_registry(broken)


def test_contract_without_declared_use_is_rejected() -> None:
    registry = load_registry(REGISTRY_PATH)
    broken = json.loads(json.dumps(registry))
    scene = rule_by_id(broken, "audited_scene_queries_v1")
    scene["shadow_lf_vote_contract"] = rule_by_id(
        broken, "witnessed_clipwide_intervention_scan_v1"
    )["shadow_lf_vote_contract"]
    with pytest.raises(ValueError, match="without declared use"):
        validate_registry(broken)


def test_failed_transfer_rule_cannot_declare_shadow_vote() -> None:
    registry = load_registry(REGISTRY_PATH)
    broken = json.loads(json.dumps(registry))
    failed = rule_by_id(broken, "instructional_v23_qwen_gemma_consensus")
    failed["allowed_uses"] = ["shadow_lf_vote"]
    with pytest.raises(ValueError, match="failed rule cannot have allowed uses"):
        validate_registry(broken)
