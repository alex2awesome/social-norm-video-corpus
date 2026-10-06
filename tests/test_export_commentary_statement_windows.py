from scripts.export_commentary_statement_windows import (
    overlapping_segments,
    statement_window,
)


def test_statement_window_clamps_to_source():
    assert statement_window(2, 4, 10, 5, 8) == (0.0, 10)


def test_overlapping_segments_includes_boundary_context():
    transcript = {
        "segments": [
            {"start": 0, "end": 1, "text": "before"},
            {"start": 2, "end": 3, "text": "inside"},
            {"start": 4, "end": 5, "text": "after"},
        ]
    }
    rows = overlapping_segments(transcript, 1, 4)
    assert [row["text"] for row in rows] == ["before", "inside", "after"]
