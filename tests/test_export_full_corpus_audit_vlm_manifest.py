from scripts.export_full_corpus_audit_vlm_manifest import clip_relative_transcript


def test_clip_relative_transcript_offsets_witnessed_source_times():
    packet = {
        "clip_window": [100.0, 120.0],
        "transcript_segments": [
            {"start": 99.0, "end": 102.0, "text": "before and during"},
            {"start": 105.0, "end": 106.0, "text": "inside"},
        ],
    }
    assert clip_relative_transcript(packet) == [
        {"start": 0.0, "end": 2.0, "text": "before and during"},
        {"start": 5.0, "end": 6.0, "text": "inside"},
    ]


def test_clip_relative_transcript_preserves_non_witnessed_times():
    packet = {
        "transcript_segments": [
            {"start": 1.5, "end": 2.5, "text": "hello"},
        ]
    }
    assert clip_relative_transcript(packet) == [
        {"start": 1.5, "end": 2.5, "text": "hello"},
    ]
