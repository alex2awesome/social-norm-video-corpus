import hashlib
import json
from pathlib import Path

import pytest

from scripts.package_blind_clips import package


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_packages_only_requested_opaque_clips(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    blind = tmp_path / "blind.jsonl"
    write_jsonl(
        blind,
        [
            {
                "audit_index": 0,
                "candidate_id": "opaque-0",
                "sheet_path": "hidden.jpg",
                "sheet_sha256": "a" * 64,
            }
        ],
    )
    semantic = tmp_path / "sealed.jsonl"
    write_jsonl(
        semantic,
        [
            {
                "audit_index": 0,
                "candidate_id": "opaque-0",
                "source_clip": str(source),
                "source_clip_sha256": digest(source),
                "norm": "must remain sealed",
            }
        ],
    )
    rows = package(blind, semantic, tmp_path / "out", {0})
    assert set(rows[0]) == {
        "audit_index",
        "candidate_id",
        "clip_path",
        "clip_sha256",
    }
    assert Path(rows[0]["clip_path"]).read_bytes() == b"video"
    assert "norm" not in rows[0]


def test_rejects_semantic_leak_in_blind_manifest(tmp_path: Path) -> None:
    blind = tmp_path / "blind.jsonl"
    write_jsonl(
        blind,
        [
            {
                "audit_index": 0,
                "candidate_id": "opaque-0",
                "sheet_path": "hidden.jpg",
                "sheet_sha256": "a" * 64,
                "norm": "leak",
            }
        ],
    )
    semantic = tmp_path / "sealed.jsonl"
    write_jsonl(semantic, [{"candidate_id": "opaque-0"}])
    with pytest.raises(ValueError, match="semantic"):
        package(blind, semantic, tmp_path / "out", {0})
