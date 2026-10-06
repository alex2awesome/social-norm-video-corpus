from scripts.recover_failed_dense_sheets_ffmpeg import failed_rows


def test_failed_rows_keeps_every_explicit_failure() -> None:
    rows = [
        {"candidate_id": "a", "error": None},
        {"candidate_id": "b", "error": "no frames"},
        {"candidate_id": "c"},
        {"candidate_id": "d", "error": "decode"},
    ]
    assert [row["candidate_id"] for row in failed_rows(rows)] == ["b", "d"]
