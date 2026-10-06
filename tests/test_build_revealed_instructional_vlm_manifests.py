from pathlib import Path

import pytest

from scripts.build_revealed_instructional_vlm_manifests import build


def test_build_orders_rows_and_resolves_storyboards(tmp_path: Path) -> None:
    rows = [
        {
            "audit_index": 1,
            "item_id": "i1",
            "uid": "u1",
            "norm": "sharing",
            "storyboard": {"item_id": "i1", "sheet_path": "storyboards/1.jpg"},
        },
        {
            "audit_index": 0,
            "item_id": "i0",
            "uid": "u0",
            "norm": "waiting",
            "storyboard": {"item_id": "i0", "sheet_path": "storyboards/0.jpg"},
        },
    ]
    storyboards, metadata = build(rows, tmp_path)
    assert [row["item_id"] for row in storyboards] == ["i0", "i1"]
    assert storyboards[0]["sheet_path"] == str(
        (tmp_path / "storyboards/0.jpg").resolve()
    )
    assert [row["norm"] for row in metadata] == ["waiting", "sharing"]


def test_build_rejects_duplicate_items(tmp_path: Path) -> None:
    rows = [
        {"audit_index": 0, "item_id": "i", "storyboard": {"sheet_path": "a.jpg"}},
        {"audit_index": 1, "item_id": "i", "storyboard": {"sheet_path": "b.jpg"}},
    ]
    with pytest.raises(ValueError, match="duplicate item_id"):
        build(rows, tmp_path)
