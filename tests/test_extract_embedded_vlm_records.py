import json

import pytest

from scripts.extract_embedded_vlm_records import extract


def test_extracts_matching_records(tmp_path) -> None:
    source = tmp_path / "selection.jsonl"
    source.write_text(
        json.dumps(
            {
                "audit_index": 0,
                "item_id": "item",
                "v10a": {"item_id": "item", "result": {"demo_usable": "yes"}},
            }
        )
        + "\n"
    )
    assert extract(source, "v10a")[0]["item_id"] == "item"


def test_rejects_identity_mismatch(tmp_path) -> None:
    source = tmp_path / "selection.jsonl"
    source.write_text(
        json.dumps(
            {
                "audit_index": 0,
                "item_id": "item",
                "v10a": {"item_id": "other"},
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="mismatch"):
        extract(source, "v10a")
