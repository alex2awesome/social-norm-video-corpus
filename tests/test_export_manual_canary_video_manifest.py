import pytest

from scripts.export_manual_canary_video_manifest import build_records


def test_build_records_separates_strict_and_relabel_gold():
    vlm = [
        {
            "item_id": "instructional:x:0",
            "uid": "x",
            "ordinal": 0,
            "pillar": "instructional",
        }
    ]
    manual = [
        {
            "item_id": "instructional:x:0",
            "uid": "x",
            "ordinal": 0,
            "decision": "accept_after_relabel",
            "evidence": "acted scene",
            "visual_demo_present": "yes",
            "is_social_norm": "yes",
            "norm_supported": "no",
        }
    ]
    proxies = {
        "records": [
            {
                "item_id": "instructional:x:0",
                "uid": "x",
                "ordinal": 0,
                "source_clip_path": "/source.mp4",
                "proxy_path": "/proxy.mp4",
                "proxy_sha256": "abc",
            }
        ]
    }

    record = build_records(vlm, manual, proxies)[0]

    assert record["gold_keep_after_relabel"] is True
    assert record["gold_label_matched_visible"] is False
    assert record["gold_usable"] is False


def test_build_records_rejects_incomplete_proxy_coverage():
    with pytest.raises(ValueError, match="cover the same items"):
        build_records(
            [{"item_id": "instructional:x:0", "uid": "x", "ordinal": 0}],
            [
                {
                    "item_id": "instructional:x:0",
                    "uid": "x",
                    "ordinal": 0,
                }
            ],
            {"records": []},
        )
