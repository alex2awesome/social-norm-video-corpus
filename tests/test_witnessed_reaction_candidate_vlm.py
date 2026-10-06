import json

from scripts.render_witnessed_reaction_candidate_storyboards import candidate_rows
from scripts.evaluate_witnessed_reaction_candidate_vlm import (
    candidate_positive,
    clip_predictions,
)
from scripts.run_witnessed_reaction_candidate_vlm import parse_result
from scripts.run_witnessed_reaction_candidate_visual_vlm import (
    parse_result as parse_visual_result,
)


def valid_result():
    return {
        "candidate_grounded": "yes",
        "triggering_action_visible_or_audible": "yes",
        "action_before_or_overlaps_candidate": "yes",
        "reaction_targets_action": "yes",
        "reaction_source_role": "bystander",
        "reaction_content": "targeted_objection",
        "reactor_visibly_distinct_from_actor": "yes",
        "strict_bystander_reaction": "yes",
        "evidence": "A second person tells the actor to stop.",
    }


def valid_visual_result():
    return {
        "triggering_action_visible": "yes",
        "visible_response_at_candidate_moment": "yes",
        "response_after_or_overlaps_action": "yes",
        "visibly_distinct_third_person_responds": "yes",
        "visible_response_targets_action": "yes",
        "visual_responder_role": "bystander",
        "visible_response_content": "interposition_or_separation",
        "strict_visual_bystander_response": "yes",
        "evidence": "A third person steps between two people.",
    }


def test_candidate_rows_joins_clip_and_excludes_negative_candidates():
    proposals = [
        {
            "item_id": "witnessed:uid:0",
            "uid": "uid",
            "candidates": [
                {
                    "segment_index": 2,
                    "text": "Stop.",
                    "start": 3,
                    "end": 4,
                    "window_start": 0,
                    "window_end": 7,
                    "mechanisms": ["direct"],
                    "negative_self_defense": False,
                    "negative_reported": False,
                },
                {
                    "segment_index": 3,
                    "text": "Get away from me.",
                    "start": 5,
                    "end": 6,
                    "window_start": 0,
                    "window_end": 9,
                    "mechanisms": ["direct"],
                    "negative_self_defense": True,
                    "negative_reported": False,
                },
            ],
        }
    ]
    rows = candidate_rows(
        proposals,
        [{"item_id": "witnessed:uid:0", "proxy_clip": "clip.mp4"}],
    )
    assert len(rows) == 1
    assert rows[0]["candidate_id"] == "witnessed:uid:0:candidate_2"


def test_parse_result_accepts_consistent_strict_positive():
    assert parse_result(json.dumps(valid_result()))["strict_bystander_reaction"] == "yes"


def test_parse_result_repairs_inconsistent_positive():
    row = valid_result()
    row["reaction_source_role"] = "affected_target"
    result = parse_result(json.dumps(row))
    assert result["strict_bystander_reaction"] == "uncertain"
    assert result["consistency_repair"] == "inconsistent_positive_to_uncertain"


def test_visual_parser_accepts_consistent_positive():
    result = parse_visual_result(json.dumps(valid_visual_result()))
    assert result["strict_visual_bystander_response"] == "yes"


def test_visual_parser_repairs_role_inconsistency():
    row = valid_visual_result()
    row["visual_responder_role"] = "affected_target"
    result = parse_visual_result(json.dumps(row))
    assert result["strict_visual_bystander_response"] == "uncertain"


def test_candidate_positive_reports_authority_separately():
    row = valid_result()
    row["reaction_source_role"] = "authority_or_host"
    assert not candidate_positive(row, include_authority=False)
    assert candidate_positive(row, include_authority=True)


def test_candidate_positive_understands_visual_only_schema():
    row = valid_visual_result()
    assert candidate_positive(row, include_authority=False)
    row["visual_responder_role"] = "authority_or_host"
    assert not candidate_positive(row, include_authority=False)
    assert candidate_positive(row, include_authority=True)


def test_visual_soft_diagnostic_may_abstain_on_order_but_strict_cannot():
    row = valid_visual_result()
    row["response_after_or_overlaps_action"] = "no"
    assert not candidate_positive(row, include_authority=False)
    assert candidate_positive(
        row,
        include_authority=False,
        allow_unresolved_visual_order=True,
    )


def test_clip_prediction_is_any_positive_candidate():
    positive = valid_result()
    negative = valid_result()
    negative["candidate_grounded"] = "no"
    rows = [
        {"item_id": "a", "result": negative, "error": None},
        {"item_id": "a", "result": positive, "error": None},
        {"item_id": "b", "result": positive, "error": "failed"},
    ]
    assert clip_predictions(rows, include_authority=False) == {
        "a": True,
        "b": False,
    }
