import pytest

from scripts.build_instructional_v20_population_manifest import build


def test_build_excludes_every_audited_uid_and_preserves_hash() -> None:
    source = [
        {
            "item_id": "instructional:a:0",
            "uid": "a",
            "source_clip": "/a.mp4",
            "source_clip_sha256": "aaa",
        },
        {
            "item_id": "instructional:b:0",
            "uid": "b",
            "source_clip": "/b.mp4",
            "source_clip_sha256": "bbb",
        },
    ]
    rows = build(source, [{"uid": "a"}])
    assert [row["uid"] for row in rows] == ["b"]
    assert rows[0]["source_sha256"] == "bbb"
    assert rows[0]["shadow_population"] == "v20_source_disjoint_unaudited"


def test_build_refuses_duplicate_source_uids() -> None:
    with pytest.raises(ValueError, match="duplicate source UID"):
        build(
            [
                {"item_id": "a:0", "uid": "a"},
                {"item_id": "a:1", "uid": "a"},
            ],
            [],
        )
