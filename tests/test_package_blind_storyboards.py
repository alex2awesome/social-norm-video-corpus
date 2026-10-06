import hashlib
import json
from pathlib import Path

import pytest

from scripts.package_blind_storyboards import package


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_package_validates_hashes_and_emits_only_blind_fields(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"pixels")
    manifest = tmp_path / "blind.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audit_index": 0,
                "candidate_id": "opaque-0",
                "sheet_path": str(source),
                "sheet_sha256": digest(source),
            }
        )
        + "\n"
    )
    rows = package(manifest, tmp_path / "packaged")
    assert set(rows[0]) == {
        "audit_index",
        "candidate_id",
        "sheet_path",
        "sheet_sha256",
    }
    assert Path(rows[0]["sheet_path"]).read_bytes() == b"pixels"


def test_package_rejects_semantic_leak(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"pixels")
    manifest = tmp_path / "blind.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audit_index": 0,
                "candidate_id": "opaque-0",
                "sheet_path": str(source),
                "sheet_sha256": digest(source),
                "norm": "secret",
            }
        )
        + "\n"
    )
    with pytest.raises(ValueError, match="semantic"):
        package(manifest, tmp_path / "packaged")


def test_package_resolves_relative_sheet_from_explicit_source_root(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "frozen_source"
    source_root.mkdir()
    source = source_root / "storyboards" / "source.jpg"
    source.parent.mkdir()
    source.write_bytes(b"pixels")
    manifest_dir = tmp_path / "new_audit"
    manifest_dir.mkdir()
    manifest = manifest_dir / "blind.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "audit_index": 0,
                "candidate_id": "opaque-0",
                "sheet_path": "storyboards/source.jpg",
                "sheet_sha256": digest(source),
            }
        )
        + "\n"
    )
    rows = package(manifest, tmp_path / "packaged", source_root)
    assert Path(rows[0]["sheet_path"]).read_bytes() == b"pixels"
