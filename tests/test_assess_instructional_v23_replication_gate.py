from scripts.assess_instructional_v23_replication_gate import assess


def prereg() -> dict:
    return {"maintenance_gate": {
        "required_items": 2,
        "retain_candidate_generation_and_review_ranking": {
            "minimum_precision": 0.75, "minimum_recall": 0.25, "minimum_selected": 2,
        },
        "downgrade_to_review_ranking_only": {
            "minimum_precision": 0.60, "minimum_recall": 0.15, "minimum_selected": 1,
        },
    }}


def summary(precision: float, recall: float, selected: int) -> dict:
    return {
        "items": 2,
        "manual_coverage_complete": True,
        "manual_visual_demos": 1,
        "manual_usable_demos": 1,
        "model_failed_items": {
            "visual": {"q": [], "g": []},
            "semantic": {"q": [], "g": []},
        },
        "pipeline_consensus": {
            "precision": precision, "recall": recall, "selected": selected,
        },
    }


def reviews(complete: bool = True) -> list[dict[str, str]]:
    value = "Y" if complete else "N"
    return [{
        "audit_index": str(i),
        "qwen_visual_reviewed": value,
        "qwen_semantic_reviewed": "Y",
        "gemma_visual_reviewed": "Y",
        "gemma_semantic_reviewed": "Y",
    } for i in range(2)]


def test_retain_gate() -> None:
    result = assess(summary(0.8, 0.5, 2), prereg(), reviews())
    assert result["maintenance_action"] == "retain_candidate_generation_and_review_ranking"


def test_downgrade_gate() -> None:
    result = assess(summary(0.65, 0.5, 1), prereg(), reviews())
    assert result["maintenance_action"] == "downgrade_to_review_ranking_only"


def test_incomplete_manual_review_disables() -> None:
    result = assess(summary(1.0, 1.0, 2), prereg(), reviews(False))
    assert result["maintenance_action"] == "disable_rule"
    assert result["model_output_reviews_complete"] is False
