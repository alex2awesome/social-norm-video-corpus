from scripts.select_instructional_v9_corpus_validation import (
    blind_rows,
    positive,
)


def visual(event: str = "yes") -> dict:
    return {
        "result": {
            "observable_event": event,
            "literal_actor": "person",
            "literal_action_or_situated_utterance": "acts",
            "literal_affected_party_or_shared_setting": "other",
            "event_start_percent": 10,
            "event_end_percent": 80,
            "same_event_actor_action_target": "yes",
            "event_temporally_localized": "yes",
            "evidence_source": "physical_action",
            "scene_role": "demonstrated_event",
            "demonstration_kind": "interpersonal_conduct",
        }
    }


def test_positive_rechecks_visual_contract():
    row = {
        "visual": visual(),
        "alignment": {"result": {"usable_after_relabel": "yes"}},
    }
    assert positive(row)
    row["visual"]["result"]["scene_role"] = "generic_context"
    assert not positive(row)


def test_blind_rows_strip_semantics_and_scores():
    source = {
        "audit_index": 0,
        "candidate_id": "opaque",
        "item_id": "instructional:x:0",
        "uid": "x",
        "pillar": "instructional",
        "band": "secret",
        "norm": "secret norm",
        "polarity": "violation",
        "storyboard_path": "/s/1.jpg",
        "storyboard_sha256": "abc",
        "v9a": {"result": {}},
        "v9b": {"result": {}},
    }
    row = blind_rows([source])[0]
    assert "band" not in row
    assert "norm" not in row
    assert "polarity" not in row
    assert "v9a" not in row
    assert "v9b" not in row
