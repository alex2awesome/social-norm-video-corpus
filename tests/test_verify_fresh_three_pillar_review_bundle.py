import json
from pathlib import Path

import pytest

from scripts.package_fresh_three_pillar_review_bundle import package
from scripts.verify_fresh_three_pillar_review_bundle import verify
from tests.test_package_fresh_three_pillar_review_bundle import build_complete_batch


def test_verify_rejects_portable_path_escape(tmp_path: Path):
    stage, run, scores = build_complete_batch(tmp_path)
    bundle = tmp_path / "bundle"
    package(stage, run, scores, bundle)
    manifest = bundle / "metadata/instructional/storyboard_manifest.jsonl"
    row = json.loads(manifest.read_text())
    row["sheet_path"] = "../../escape.jpg"
    manifest.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="escapes root"):
        verify(bundle)


def test_verify_rejects_unreferenced_audio(tmp_path: Path):
    stage, run, scores = build_complete_batch(tmp_path)
    bundle = tmp_path / "bundle"
    package(stage, run, scores, bundle)
    extra = bundle / "audio/extra.ogg"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"not referenced")
    with pytest.raises(ValueError, match="unreferenced audio"):
        verify(bundle)
