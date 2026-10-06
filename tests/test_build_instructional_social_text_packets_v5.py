import pytest

from scripts.build_instructional_social_text_packets_v5 import FORBIDDEN, build


def test_builds_label_only_packet_without_retrieval_or_visual_leakage():
    rows = [{
        "audit_index": 0,
        "item_id": "instructional:x:0",
        "uid": "x",
        "title": "leaky",
        "found_by_query": "leaky query",
        "norm": "respect boundaries",
        "polarity": "violation",
        "start_quote": "stop touching my dog",
        "end_quote": "leave him alone",
        "explanation": "",
    }]
    packets = build(rows)
    assert len(packets) == 1
    assert not (set(packets[0]) & FORBIDDEN)
    assert packets[0]["proposed_norm"] == "respect boundaries"


def test_rejects_reordered_or_incomplete_indices():
    with pytest.raises(ValueError, match="indices"):
        build([{"audit_index": 1}])
