import json

import pytest

from scripts.export_commentary_temporal_model_manifest import (
    export_manifest,
    sanitize_row,
)


def masked_row():
    return {
        "audit_index": 1,
        "ocr_masked": True,
        "page_paths": ["pages_ocr_masked/0001/page_01.jpg"],
        "page_sha256": ["a" * 64],
        "human_unmasked_page_paths": [
            "pages_unmasked_human_only/0001/page_01.jpg"
        ],
        "human_unmasked_page_sha256": ["b" * 64],
    }


def test_sanitize_removes_all_human_only_fields():
    row = sanitize_row(masked_row())
    assert "human_unmasked_page_paths" not in row
    assert "human_unmasked_page_sha256" not in row
    assert "unmasked" not in json.dumps(row).lower()


@pytest.mark.parametrize(
    "page_path",
    [
        "pages_unmasked_human_only/0001/page_01.jpg",
        "../pages_ocr_masked/0001/page_01.jpg",
        "/tmp/pages_ocr_masked/0001/page_01.jpg",
    ],
)
def test_sanitize_rejects_unsafe_or_nonmasked_page_roots(page_path):
    row = masked_row()
    row["page_paths"] = [page_path]
    with pytest.raises(ValueError):
        sanitize_row(row)


def test_sanitize_requires_explicit_mask_flag():
    row = masked_row()
    row["ocr_masked"] = False
    with pytest.raises(ValueError):
        sanitize_row(row)


def test_export_refuses_to_overwrite(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(masked_row()) + "\n")
    destination = tmp_path / "model.jsonl"
    assert export_manifest(source, destination) == 1
    with pytest.raises(FileExistsError):
        export_manifest(source, destination)
