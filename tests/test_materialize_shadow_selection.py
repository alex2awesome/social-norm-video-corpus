import pytest

from scripts.materialize_shadow_selection import materialize


def test_materialize_preserves_manifest_and_attaches_selection_fields():
    manifest = [
        {"item_id": "a", "source_clip": "/a.mp4", "norm": "respect"},
        {"item_id": "b", "source_clip": "/b.mp4", "norm": "quiet"},
    ]
    selection = [{"item_id": "b", "score_band": "high", "norm": "quiet"}]
    assert materialize(manifest, selection) == [
        {
            "item_id": "b",
            "source_clip": "/b.mp4",
            "norm": "quiet",
            "shadow_selection": {"score_band": "high"},
        }
    ]


def test_materialize_fails_on_missing_or_duplicate_selection():
    manifest = [{"item_id": "a"}]
    with pytest.raises(ValueError, match="absent"):
        materialize(manifest, [{"item_id": "b"}])
    with pytest.raises(ValueError, match="unique"):
        materialize(manifest, [{"item_id": "a"}, {"item_id": "a"}])
