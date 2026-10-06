from __future__ import annotations

from scripts.evaluate_witnessed_authority_cues import evaluate


def test_evaluate_reports_precision_recall_and_visual_counts():
    sealed = [
        {"audit_index": 0, "item_id": "a", "uid": "u1"},
        {"audit_index": 1, "item_id": "b", "uid": "u2"},
        {"audit_index": 2, "item_id": "c", "uid": "u3"},
        {"audit_index": 3, "item_id": "d", "uid": "u4"},
    ]
    post = [
        {"audit_index": 0, "authority_or_host_reaction": "yes", "reason": "x"},
        {"audit_index": 1, "authority_or_host_reaction": "yes", "reason": "x"},
        {"audit_index": 2, "authority_or_host_reaction": "no", "reason": "x"},
        {"audit_index": 3, "authority_or_host_reaction": "no", "reason": "x"},
    ]
    blind = [
        {"audit_index": 0, "visual_scene_candidate": "yes"},
        {"audit_index": 1, "visual_scene_candidate": "uncertain"},
        {"audit_index": 2, "visual_scene_candidate": "no"},
        {"audit_index": 3, "visual_scene_candidate": "yes"},
    ]
    scores = [
        {"item_id": "a", "authority_reaction_cue": True},
        {"item_id": "b", "authority_reaction_cue": False},
        {"item_id": "c", "authority_reaction_cue": True},
        {"item_id": "d", "authority_reaction_cue": False},
    ]
    result = evaluate(sealed, post, blind, {"v": scores})
    assert result["gold_authority_or_host_reactions"] == 2
    assert result["blind_visual_scene_counts"] == {
        "yes": 2,
        "uncertain": 1,
        "no": 1,
    }
    assert result["models"]["v"]["precision"] == 0.5
    assert result["models"]["v"]["recall"] == 0.5
