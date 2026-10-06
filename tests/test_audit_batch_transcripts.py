import pytest

from scripts.export_audit_batch_transcripts import build_records
from scripts.run_ollama_transcript_audit import (
    index_transcripts,
    transcript_for_row,
)


def test_export_preserves_item_local_transcripts_for_same_uid():
    batch = {
        "items": [
            {
                "ordinal": 0,
                "uid": "x",
                "aligned_transcript": [
                    {
                        "clip_start": -0.5,
                        "clip_end": 1.0,
                        "text": " first ",
                    }
                ],
            },
            {
                "ordinal": 1,
                "uid": "x",
                "aligned_transcript": [
                    {"clip_start": 2.0, "clip_end": 3.0, "text": "second"}
                ],
            },
        ]
    }
    vlm = [
        {"ordinal": 0, "uid": "x", "item_id": "instructional:x:0"},
        {"ordinal": 1, "uid": "x", "item_id": "instructional:x:1"},
    ]

    records = build_records(batch, vlm)
    by_item, by_uid = index_transcripts(records)

    assert not by_uid
    assert transcript_for_row(vlm[0], by_item, by_uid)["segments"] == [
        {"start": 0.0, "end": 1.0, "text": "first"}
    ]
    assert transcript_for_row(vlm[1], by_item, by_uid)["segments"][0][
        "text"
    ] == "second"


def test_transcript_index_rejects_duplicate_item_ids():
    row = {"item_id": "instructional:x:0", "uid": "x", "segments": []}

    with pytest.raises(ValueError, match="duplicate transcript item_id"):
        index_transcripts([row, row])


def test_transcript_lookup_falls_back_to_legacy_uid():
    legacy = {"uid": "x", "segments": []}
    by_item, by_uid = index_transcripts([legacy])

    assert transcript_for_row(
        {"item_id": "instructional:x:0", "uid": "x"},
        by_item,
        by_uid,
    ) is legacy
