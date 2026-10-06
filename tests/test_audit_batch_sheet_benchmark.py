import hashlib
import json
from pathlib import Path

import pytest

from export_audit_batch_sheet_benchmark import build_records


def test_build_records_checks_identity_and_hashes(tmp_path: Path):
    (tmp_path / "sheets").mkdir()
    sheet = tmp_path / "sheets" / "0000_uid_p00.jpg"
    sheet.write_bytes(b"image")
    digest = hashlib.sha256(b"image").hexdigest()
    source = {
        "items": [
            {
                "item_id": "instructional:uid:0",
                "ordinal": 0,
                "uid": "uid",
                "pillar": "instructional",
                "category": "instr_family",
                "polarity": "violation",
                "found_by_query": "family conflict role play",
                "norm": "do not insult",
                "explanation": "One person insults another.",
                "frame_manifest_sha256": "frames",
            }
        ]
    }
    sheets = {
        "items": [
            {
                "item_id": "instructional:uid:0",
                "frame_manifest_sha256": "frames",
                "pages": [{"path": "sheets/0000_uid_p00.jpg", "sha256": digest}],
            }
        ]
    }
    (tmp_path / "manifest.json").write_text(json.dumps(source))
    (tmp_path / "sheets" / "manifest.json").write_text(json.dumps(sheets))

    records = build_records(tmp_path)

    assert records[0]["query"].startswith("do not insult; detector explanation:")
    assert records[0]["gold_target_source_pass"] is None
    assert records[0]["sheet_sha256"] == digest


def test_build_records_rejects_hash_mismatch(tmp_path: Path):
    (tmp_path / "sheets").mkdir()
    (tmp_path / "sheets" / "x.jpg").write_bytes(b"image")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "item_id": "x",
                        "ordinal": 0,
                        "uid": "u",
                        "pillar": "instructional",
                        "norm": "n",
                        "frame_manifest_sha256": "frames",
                    }
                ]
            }
        )
    )
    (tmp_path / "sheets" / "manifest.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "item_id": "x",
                        "frame_manifest_sha256": "frames",
                        "pages": [{"path": "sheets/x.jpg", "sha256": "wrong"}],
                    }
                ]
            }
        )
    )

    with pytest.raises(ValueError, match="sheet hash mismatch"):
        build_records(tmp_path)
