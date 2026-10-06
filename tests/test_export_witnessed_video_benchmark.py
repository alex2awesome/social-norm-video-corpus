import pytest

from scripts.export_witnessed_video_benchmark import build_records


def inputs(disposition="recover_strict"):
    source = {
        "items": [
            {
                "item_id": "witnessed:x:0",
                "ordinal": 0,
                "uid": "x",
                "category": "c",
                "norm": "n",
                "explanation": "e",
            }
        ]
    }
    manual = [
        {
            "item_id": "witnessed:x:0",
            "uid": "x",
            "disposition": disposition,
            "authenticity": "organic",
            "description": "visible action then objection",
        }
    ]
    proxies = {
        "records": [
            {
                "item_id": "witnessed:x:0",
                "ordinal": 0,
                "uid": "x",
                "source_clip_path": "/source.mp4",
                "proxy_path": "/proxy.mp4",
                "proxy_sha256": "abc",
            }
        ]
    }
    return source, manual, proxies


def test_recovery_gold_is_separate_from_current_label_acceptance():
    record = build_records(*inputs())[0]
    assert record["gold_strict_current"] is False
    assert record["gold_witnessed_recovery"] is True
    assert record["gold_instructional_reroute"] is False
    assert record["gold_any_visual_recovery"] is True


def test_instructional_reroute_is_preserved_as_visual_recovery():
    record = build_records(*inputs("reroute_instructional_review"))[0]
    assert record["gold_witnessed_recovery"] is False
    assert record["gold_instructional_reroute"] is True
    assert record["gold_any_visual_recovery"] is True


def test_incomplete_coverage_is_rejected():
    source, manual, proxies = inputs()
    proxies["records"] = []
    with pytest.raises(ValueError, match="cover the same items"):
        build_records(source, manual, proxies)
