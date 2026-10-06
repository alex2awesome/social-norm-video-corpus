from scripts.evaluate_witnessed_speaker_embedding_proxy import (
    candidate_rules,
    evaluate,
    metrics,
    validate_candidate_ledger,
)


def voice(candidate: str, item: str, differs: int, new: int, sandwich: int) -> dict:
    return {
        "candidate_id": candidate,
        "item_id": item,
        "error": None,
        "features": {
            "speaker_proxy.differs_from_immediate_before": differs,
            "speaker_proxy.new_vs_recent_prior": new,
            "speaker_proxy.sandwiched_third_voice": sandwich,
        },
    }


def manual(item: str, role: str) -> dict:
    return {
        "item_id": item,
        "reaction_source_role": role,
        "reaction_grounding": "visible_on_scene",
        "reaction_content": "targeted_objection",
        "temporal_relation": "action_established_before_reaction",
        "description": "manual",
    }


def test_candidate_rules_expose_proxy_separately_from_visual_intersection() -> None:
    rules = candidate_rules(voice("c", "i", 1, 1, 0), None)
    assert rules["voice_differs_immediate_before"]
    assert not rules["qwen_visual_and_voice_differs"]


def test_evaluate_aggregates_any_candidate_at_clip_level() -> None:
    audited, report = evaluate(
        [manual("positive", "bystander"), manual("negative", "affected_target")],
        [
            voice("c1", "positive", 0, 0, 0),
            voice("c2", "positive", 1, 1, 0),
            voice("c3", "negative", 0, 0, 0),
        ],
        [],
    )
    assert len(audited) == 3
    result = report["rules"]["voice_differs_immediate_before"]["strict_bystander"]
    assert result["precision"] == 1
    assert result["recall"] == 1


def test_metrics_counts_missing_predictions_as_negative() -> None:
    result = metrics({"a": True, "b": False}, {})
    assert result["fn"] == 1
    assert result["tn"] == 1


def test_manual_candidate_ledger_requires_exact_coverage() -> None:
    rows = [voice("c", "i", 1, 1, 0)]
    ledger = [
        {
            "candidate_id": "c",
            "visual_identity_role": "bystander",
            "speaker_proxy_disposition": "candidate_generation_only",
        }
    ]
    assert validate_candidate_ledger(rows, ledger)[
        "manual_candidate_coverage_complete"
    ]
