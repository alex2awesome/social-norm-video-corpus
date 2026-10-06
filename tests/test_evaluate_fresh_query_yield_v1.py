from scripts.evaluate_fresh_query_yield_v1 import evaluate


def reaction(staging="no_clear_staging_evidence"):
    return {
        "reaction_grounded": "yes",
        "action_before_or_overlaps_response": "yes",
        "response_targets_action": "yes",
        "responder_role": "separate_bystander",
        "response_content": "targeted_objection",
        "trigger_kind": "interpersonal_treatment",
        "staging": staging,
        "source_audio_review_status": "reviewed",
        "speaker_identity_basis": "voice_and_visible_turn_binding",
        "manual_evidence": "A visible third party objects after the act.",
    }


def test_search_yield_keeps_pillar_targets_separate():
    provenance = [
        {"pillar": "instructional", "uid": "i", "query": "role play", "query_source": "instructional"},
        {"pillar": "witnessed", "uid": "w", "query": "bystander steps in", "query_source": "taxonomy"},
    ]
    instructional = [{"item_id": "instructional:i:0", "uid": "i"}]
    instruction_gold = [{
        "item_id": "instructional:i:0", "uid": "i", "visual_demo": "yes",
        "label_alignment": "partial",
        "source_audio_review_status": "reviewed",
        "demo_evidence_modalities": "audiovisual_other",
    }]
    witnessed = [{
        "item_id": "witnessed:w:clip_0", "uid": "w",
        "cohort": "uniform_probability_sample",
        "candidates": [{"candidate_id": "c"}],
    }]
    witnessed_gold = [{
        "candidate_id": "c", "item_id": "witnessed:w:clip_0", "uid": "w",
        **reaction("clearly_staged"),
    }]
    report = evaluate(
        provenance, instructional, instruction_gold, witnessed, witnessed_gold,
        [{"candidate_id": "c", "audio_present": True}],
        [{"item_id": "instructional:i:0", "audio_present": True}],
    )
    instruction = report["instructional"]["by_query_source"]["instructional"]
    assert instruction["clean_label_aligned_demo"] == 0
    assert instruction["demo_usable_after_relabel"] == 1
    witnessed_result = report["witnessed"]["population_query_yield"][
        "by_query_source"
    ]["taxonomy"]
    assert witnessed_result["distinct_bystander_reaction"] == 1
    assert witnessed_result["strict_organic_social_reaction"] == 0
    assert report["automatic_search_change_authorized"] is False


def test_enrichment_is_excluded_from_population_query_yield():
    provenance = [
        {"pillar": "instructional", "uid": "i", "query": "demo", "query_source": "q"},
        {"pillar": "witnessed", "uid": "w", "query": "reaction", "query_source": "q"},
    ]
    report = evaluate(
        provenance,
        [{"item_id": "i", "uid": "i"}],
        [{
            "item_id": "i", "uid": "i", "visual_demo": "no",
            "label_alignment": "no_visual",
            "source_audio_review_status": "source_has_no_audio",
            "demo_evidence_modalities": "no_demo",
        }],
        [{
            "item_id": "w", "uid": "w", "cohort": "v3_positive_enrichment",
            "candidates": [{"candidate_id": "c"}],
        }],
        [{"candidate_id": "c", "item_id": "w", "uid": "w", **reaction()}],
        [{"candidate_id": "c", "audio_present": True}],
        [{"item_id": "i", "audio_present": False}],
    )
    assert report["witnessed"]["population_query_yield"]["sources"] == 0
    assert report["witnessed"]["all_cohorts_descriptive_only"][
        "by_query_source"
    ]["q"]["distinct_bystander_reaction"] == 1


def test_conflicting_provenance_is_not_attributed_to_one_query():
    provenance = [
        {
            "pillar": "instructional", "uid": "i", "query": "chosen",
            "query_source": "chosen_source", "query_conflict": True,
            "query_source_conflict": True,
        },
        {
            "pillar": "witnessed", "uid": "w", "query": "reaction",
            "query_source": "taxonomy",
        },
    ]
    report = evaluate(
        provenance,
        [{"item_id": "i", "uid": "i"}],
        [{
            "item_id": "i", "uid": "i", "visual_demo": "no",
            "label_alignment": "no_visual",
            "source_audio_review_status": "source_has_no_audio",
            "demo_evidence_modalities": "no_demo",
        }],
        [{
            "item_id": "w", "uid": "w", "cohort": "uniform_probability_sample",
            "candidates": [{"candidate_id": "c"}],
        }],
        [{"candidate_id": "c", "item_id": "w", "uid": "w", **reaction()}],
        [{"candidate_id": "c", "audio_present": True}],
        [{"item_id": "i", "audio_present": False}],
    )
    assert "<provenance_conflict>" in report["instructional"]["by_exact_query"]
    assert "chosen" not in report["instructional"]["by_exact_query"]
