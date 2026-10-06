import json

from scripts.export_commentary_followup_transcripts import (
    select_uids,
    transcript_record,
)


def test_select_uids_returns_only_unresolved_in_audit_order() -> None:
    semantic = [
        {"candidate_id": "c1", "audit_index": 1, "uid": "u1"},
        {"candidate_id": "c0", "audit_index": 0, "uid": "u0"},
    ]
    ledger = [
        {"candidate_id": "c1", "visual_status": "U"},
        {"candidate_id": "c0", "visual_status": "N"},
    ]
    assert [row["uid"] for row in select_uids(semantic, ledger)] == ["u1"]


def test_transcript_record_keeps_timestamped_text_only(tmp_path) -> None:
    (tmp_path / "u1.json").write_text(
        json.dumps(
            {
                "language": "en",
                "segments": [
                    {"start": 1.0, "end": 2.0, "text": " event words "},
                    {"start": 2.0, "end": 3.0, "text": ""},
                ],
            }
        )
    )
    result = transcript_record(
        {
            "audit_index": 1,
            "candidate_id": "c1",
            "uid": "u1",
            "norm": "title",
        },
        tmp_path,
    )
    assert result["transcript_status"] == "available"
    assert result["segments"] == [
        {"start": 1.0, "end": 2.0, "text": "event words"}
    ]


def test_transcript_record_reports_missing(tmp_path) -> None:
    result = transcript_record(
        {
            "audit_index": 1,
            "candidate_id": "c1",
            "uid": "missing",
            "norm": "title",
        },
        tmp_path,
    )
    assert result["transcript_status"] == "missing"
    assert result["segments"] == []
