import json

from scripts.score_witnessed_reaction_candidate_windows import (
    clip_segments,
    score_metadata,
)


def test_clip_segments_adds_clip_relative_times_and_preserves_speaker():
    rows = clip_segments(
        [
            {"start": 9, "end": 10, "text": "before"},
            {"start": 12, "end": 13, "text": "Stop.", "speaker": "S1"},
        ],
        10,
        15,
    )
    assert len(rows) == 2
    assert rows[1]["clip_start"] == 2
    assert rows[1]["speaker"] == "S1"


def test_score_metadata_emits_proposal_without_acceptance(tmp_path):
    hit = tmp_path / "hits" / "uid"
    hit.mkdir(parents=True)
    transcript_dir = tmp_path / "transcripts"
    transcript_dir.mkdir()
    metadata = {
        "video_id": "uid",
        "reactions": [
            {
                "clip_idx": 0,
                "clip_window": [10, 20],
                "start": 17,
                "phrase": "I was just joking",
            }
        ],
    }
    transcript = {
        "segments": [
            {"start": 11, "end": 12, "text": "What are you doing?"},
            {"start": 13, "end": 14, "text": "Leave her alone."},
            {"start": 17, "end": 18, "text": "I was just joking"},
        ]
    }
    metadata_path = hit / "metadata.json"
    transcript_path = transcript_dir / "uid.json"
    metadata_path.write_text(json.dumps(metadata))
    transcript_path.write_text(json.dumps(transcript))

    rows = score_metadata(metadata_path, transcript_path)
    assert len(rows) == 1
    row = rows[0]
    assert row["features"]["candidate_scan.active_nonnegative"] == 2
    assert row["proposal_disposition"].startswith("send_candidate")
    assert row["acceptance_label"] is None
    assert row["corpus_disposition"] is None
    assert row["policy"] == "proposal_only_shadow_non_destructive"
