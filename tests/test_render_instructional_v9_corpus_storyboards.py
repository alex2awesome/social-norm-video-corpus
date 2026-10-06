from pathlib import Path

import pytest

from scripts.render_instructional_v9_corpus_storyboards import (
    FORBIDDEN_OUTPUT_FIELDS,
    failure_record,
    opaque_name,
    public_record,
    validate_source,
)


def test_opaque_name_contains_no_source_identity():
    assert opaque_name(17) == "000017.jpg"


def test_public_record_is_semantically_blind():
    source = {
        "item_id": "instructional:source:0",
        "uid": "source",
        "pillar": "instructional",
        "title": "secret title",
        "norm": "secret norm",
        "polarity": "violation",
    }
    record = public_record(
        source,
        0,
        Path("/tmp/000000.jpg"),
        "abc",
        {
            "sampled_frames": 36,
            "fps": 3.0,
            "sampled_timestamps": [0.0, 1.0],
        },
    )
    assert not (FORBIDDEN_OUTPUT_FIELDS & record.keys())
    assert "title" not in str(record)
    assert "secret norm" not in str(record)


def test_validate_source_rejects_duplicate_identity():
    with pytest.raises(ValueError, match="duplicate"):
        validate_source(
            [
                {"item_id": "same", "uid": "a"},
                {"item_id": "same", "uid": "b"},
            ]
        )


def test_decode_failure_is_explicit_and_semantically_blind():
    row = {
        "item_id": "commentary:source:0",
        "uid": "source",
        "pillar": "commentary",
        "title": "secret event title",
        "norm": "secret norm",
    }
    record = failure_record(row, 4, RuntimeError("no frames decoded"))
    assert record["error"] == "RuntimeError: no frames decoded"
    assert record["sheet_path"] is None
    assert not (FORBIDDEN_OUTPUT_FIELDS & record.keys())
    assert "secret" not in str(record)
