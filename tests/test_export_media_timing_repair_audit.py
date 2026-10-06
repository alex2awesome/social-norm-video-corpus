import json

from scripts.export_media_timing_repair_audit import derive_interval


def test_derive_instructional_interval_uses_demo_index_and_padding(tmp_path):
    root = tmp_path / "instructional"
    item = root / "video_a"
    item.mkdir(parents=True)
    (item / "metadata.json").write_text(
        json.dumps(
            {
                "demos": [
                    {"start": 3.0, "end": 7.0, "norm": "courtesy"},
                    {"start": 20.0, "end": 23.5, "norm": "safety"},
                ]
            }
        )
    )
    uid, start, end, label = derive_interval(
        {
            "pillar": "instructional",
            "relative_path": "video_a/demo_1.mp4",
        },
        root,
    )
    assert (uid, start, end) == ("video_a", 19.0, 24.5)
    assert label["norm"] == "safety"


def test_derive_witnessed_interval_requires_shared_clip_window(tmp_path):
    root = tmp_path / "hits"
    item = root / "video_b"
    item.mkdir(parents=True)
    (item / "metadata.json").write_text(
        json.dumps(
            {
                "reactions": [
                    {
                        "clip_idx": 2,
                        "clip_window": [10.0, 22.0],
                        "norm": "consent",
                    },
                    {
                        "clip_idx": 2,
                        "clip_window": [10.0, 22.0],
                        "norm": "boundaries",
                    },
                ]
            }
        )
    )
    uid, start, end, label = derive_interval(
        {"pillar": "witnessed", "relative_path": "video_b/clip_2.mp4"},
        root,
    )
    assert (uid, start, end) == ("video_b", 10.0, 22.0)
    assert len(label["reactions"]) == 2
