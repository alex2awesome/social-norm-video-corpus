import pytest

from scripts.evaluate_instructional_visual_title_transfer_v1 import evaluate


QWEN = "qwen-test"
GEMMA = "gemma-test"


def fixtures(items=4):
    prereg = {
        "items": items,
        "kind": "instructional_visual_title_conjunction_transfer_v1",
        "gate": {
            "minimum_rendered_items": items,
            "minimum_selected": 2,
            "minimum_visual_demo_precision": 0.5,
            "minimum_visual_demo_precision_wilson_95_lower": 0.0,
        },
    }
    sealed = [
        {
            "audit_index": i,
            "item_id": f"item-{i}",
            "uid": f"uid-{i}",
            "scene_title_candidate": i < 3,
        }
        for i in range(items)
    ]
    storyboards = [{"item_id": f"item-{i}", "frame_count": 36} for i in range(items)]
    blind = [
        {
            "audit_index": str(i),
            "render_status": "success",
            "manual_visual_demo": "yes" if i in {0, 2} else "no",
        }
        for i in range(items)
    ]
    semantic = [
        {
            "audit_index": str(i),
            "original_label_usable": "yes" if i == 0 else "no",
            "label_alignment": "exact" if i == 0 else "no",
            "scene_grounded_relabel_needed": "no",
        }
        for i in range(items)
    ]
    qwen = []
    gemma = []
    audits = []
    for i in range(items):
        for model, rows in ((QWEN, qwen), (GEMMA, gemma)):
            passed = i in {0, 1, 2}
            rows.append({
                "item_id": f"item-{i}", "model": model, "error": None,
                "result": {"demo_pass": "yes" if passed else "no"},
            })
            audits.append({
                "audit_index": str(i), "model": model,
                "output_visually_supported": "yes", "error_mechanism": "none",
            })
    return prereg, sealed, storyboards, qwen, gemma, blind, semantic, audits


def test_frozen_conjunction_is_shadow_ranking_only() -> None:
    rows, report = evaluate(*fixtures())
    assert [row["visual_title_conjunction"] for row in rows] == [True, True, True, False]
    assert report["metrics"]["visual_title_conjunction"]["tp"] == 2
    assert report["metrics"]["visual_title_conjunction"]["fp"] == 1
    assert report["preregistered_pass"] is True
    assert report["allowed_uses"] == ["candidate_generation", "manual_review_ranking"]
    assert report["automatic_acceptance"] is False
    assert report["automatic_rejection"] is False
    assert report["corpus_mutation_authorized"] is False


def test_gate_fails_minimum_selected_without_changing_authority() -> None:
    data = list(fixtures())
    data[0]["gate"]["minimum_selected"] = 4
    _, report = evaluate(*data)
    assert report["checks"]["minimum_selected"] is False
    assert report["preregistered_pass"] is False
    assert report["allowed_uses"] == []
    assert report["all_media_retained"] is True


def test_manual_output_audit_requires_exact_coverage() -> None:
    data = list(fixtures())
    data[-1] = data[-1][:-1]
    with pytest.raises(ValueError, match="does not exactly cover"):
        evaluate(*data)


def test_duplicate_source_is_rejected() -> None:
    data = list(fixtures())
    data[1][1]["uid"] = data[1][0]["uid"]
    with pytest.raises(ValueError, match="source-disjoint"):
        evaluate(*data)
