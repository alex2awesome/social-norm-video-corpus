import hashlib

import pytest

from scripts.extract_hourly_monitor_snapshot import extract


def test_extract_preserves_exact_matching_jsonl_record(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    wanted = b'{"checked_at":"b","fresh_review_queue":[{"uid":"u"}]}\n'
    path.write_bytes(
        b'{"checked_at":"a"}\n' + wanted + b'{"checked_at":"c"}\n'
    )

    payload = extract(path, "b")

    assert payload == wanted
    assert hashlib.sha256(payload).hexdigest() == hashlib.sha256(wanted).hexdigest()


def test_extract_requires_unique_timestamp(tmp_path):
    path = tmp_path / "snapshots.jsonl"
    path.write_text('{"checked_at":"a"}\n{"checked_at":"a"}\n')
    with pytest.raises(ValueError, match="found 2"):
        extract(path, "a")
