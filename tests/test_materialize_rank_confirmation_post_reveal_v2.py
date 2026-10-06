from scripts.materialize_rank_confirmation_post_reveal_v2 import (
    materialize_instructional,
    materialize_witnessed,
    quote_grounded,
)


def test_quote_grounded_requires_both_quotes():
    packet = {
        "start_quote": "Shake hands",
        "end_quote": "short and firm",
        "aligned_transcript": [{"text": "Shake hands. Keep it short and firm."}],
    }
    assert quote_grounded(packet) == "yes"
    packet["end_quote"] = "not in transcript"
    assert quote_grounded(packet) == "no"


def test_materialize_instructional_strict_pass():
    item_id = "instructional:u:0"
    packets = [
        {
            "item_id": item_id,
            "audit_index": 0,
            "start_quote": "be polite",
            "end_quote": "say please",
            "aligned_transcript": [{"text": "Be polite and say please."}],
        }
    ]
    sparse = [
        {
            "item_id": item_id,
            "dense_review_required": "no",
            "situated_social_scene": "yes",
            "concrete_behavior_visible": "yes",
        }
    ]
    compact = [
        {
            "item_id": item_id,
            "assigned_norm_is_social_norm": "yes",
            "demo_visually_matches_assigned_norm": "yes",
            "polarity_matches_depiction": "yes",
            "recoverable_route": "instructional",
            "evidence_note": "Acted request.",
        }
    ]
    row = materialize_instructional(packets, sparse, [], compact)[0]
    assert row["strict_instructional_pass"] == "yes"
    assert row["review_complete"] == "yes"


def test_materialize_instructional_absent_demo_cannot_match():
    item_id = "instructional:u:0"
    packets = [
        {
            "item_id": item_id,
            "audit_index": 0,
            "start_quote": "open settings",
            "end_quote": "move slider",
            "aligned_transcript": [{"text": "Open settings and move slider."}],
        }
    ]
    sparse = [
        {
            "item_id": item_id,
            "visual_demo_present": "no",
            "dense_followup": "no",
        }
    ]
    compact = [
        {
            "item_id": item_id,
            "assigned_norm_is_social_norm": "no",
            "demo_visually_matches_assigned_norm": "yes",
            "polarity_matches_depiction": "yes",
            "recoverable_route": "retain_non_social_demo",
            "evidence_note": "Software tutorial.",
        }
    ]
    row = materialize_instructional(packets, sparse, [], compact)[0]
    assert row["visual_demo_present"] == "no"
    assert row["demo_visually_matches_assigned_norm"] == "no"


def test_materialize_witnessed_fails_staged_clip():
    item_id = "witnessed:u:0"
    packets = [{"item_id": item_id, "audit_index": 0}]
    sparse = [{"item_id": item_id}]
    compact = [
        {
            "item_id": item_id,
            "assigned_norm_relation": "exact",
            "action_visible_before_reaction": "yes",
            "reaction_role": "affected_target",
            "authenticity": "staged",
            "recoverable_route": "instructional",
            "evidence_note": "Branded prank.",
        }
    ]
    row = materialize_witnessed(packets, sparse, compact)[0]
    assert row["strict_witnessed_pass"] == "no"
    assert row["reaction_is_organic_bystander_disapproval"] == "no"
