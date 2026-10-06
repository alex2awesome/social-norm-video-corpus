import pytest

from scripts.remap_manual_item_ids_by_uid import remap_manual_rows


def test_remap_manual_rows_preserves_labels_and_records_crosswalk():
    manual = [
        {
            "item_id": "commentary:video_a:0",
            "expected_disposition": "dense_followup_commentary",
        },
        {
            "item_id": "commentary:video_b:0",
            "expected_disposition": "text_only",
        },
    ]
    targets = [
        {"item_id": "commentary:video_a:2", "uid": "video_a"},
        {"item_id": "commentary:video_b:0", "uid": "video_b"},
    ]
    result = remap_manual_rows(manual, targets)
    assert result[0]["item_id"] == "commentary:video_a:2"
    assert result[0]["original_manual_item_id"] == "commentary:video_a:0"
    assert result[0]["expected_disposition"] == "dense_followup_commentary"
    assert "original_manual_item_id" not in result[1]


def test_remap_manual_rows_rejects_nonunique_target_uids():
    with pytest.raises(ValueError, match="duplicate UID"):
        remap_manual_rows(
            [{"item_id": "commentary:video_a:0"}],
            [
                {"item_id": "commentary:video_a:1", "uid": "video_a"},
                {"item_id": "commentary:video_a:2", "uid": "video_a"},
            ],
        )
