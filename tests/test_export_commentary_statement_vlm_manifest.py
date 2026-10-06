from scripts.export_commentary_statement_vlm_manifest import convert


def test_convert_makes_transcript_clip_relative():
    row = convert(
        {
            "audit_index": 2,
            "item_id": "commentary_statement:u:0",
            "uid": "u",
            "window_start_sec": 10,
            "window_end_sec": 20,
            "window_media": "/tmp/u.mp4",
            "statement": {"norm": "respect", "quote": "That was rude", "signal": "evaluation"},
            "transcript_segments": [{"start": 12, "end": 14, "text": "That was rude"}],
        }
    )
    assert row["pillar"] == "commentary"
    assert row["aligned_transcript"][0]["start"] == 2
    assert row["duration_hint"] == 10
