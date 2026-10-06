import hashlib
import json
from pathlib import Path

from scripts.verify_distributed_youtube_consolidation import verify


def write_ledger(path: Path, worker: str, uid: str, payload: bytes) -> None:
    row = {
        "status": "downloaded",
        "worker": worker,
        "uid": uid,
        "path": f"{uid}.mp4",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    path.write_text(json.dumps(row) + "\n")


def test_verification_accepts_only_matching_destination_bytes(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    payload = b"verified media"
    uid = "youtube__abc"
    ledger = tmp_path / "sk1.jsonl"
    write_ledger(ledger, "sk1", uid, payload)
    (video_dir / f"{uid}.mp4").write_bytes(payload)

    rows, summary = verify(video_dir, [ledger])
    assert [row["uid"] for row in rows] == [uid]
    assert summary["complete"] is True
    assert summary["verified_uids"] == 1


def test_verification_reports_non_overwriting_collision(tmp_path):
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    uid = "youtube__abc"
    ledger = tmp_path / "sk2.jsonl"
    write_ledger(ledger, "sk2", uid, b"expected")
    (video_dir / f"{uid}.mp4").write_bytes(b"different")

    rows, summary = verify(video_dir, [ledger])
    assert rows == []
    assert summary["complete"] is False
    assert summary["size_mismatches"][0]["uid"] == uid
