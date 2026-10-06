import csv
from pathlib import Path

import pytest

from scripts.upgrade_fresh_manual_templates_v2 import upgrade


def write_tsv(path: Path) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["audit_index", "item_id", "uid", "visual_demo"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerow({"audit_index": "0", "item_id": "i", "uid": "u", "visual_demo": ""})


def test_upgrade_adds_blank_relabel_field_without_filling_judgment(tmp_path: Path):
    source, out = tmp_path / "source.tsv", tmp_path / "out.tsv"
    write_tsv(source)
    report = upgrade("instructional", source, out)
    with out.open(newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["corrected_behavior_label"] == ""
    assert row["source_audio_review_status"] == ""
    assert row["demo_evidence_modalities"] == ""
    assert report["prior_judgments_present"] is False


def test_upgrade_refuses_second_schema_mutation(tmp_path: Path):
    source, first = tmp_path / "source.tsv", tmp_path / "first.tsv"
    write_tsv(source)
    upgrade("instructional", source, first)
    with pytest.raises(ValueError, match="already upgraded"):
        upgrade("instructional", first, tmp_path / "second.tsv")


def test_witnessed_upgrade_adds_explicit_audio_and_identity_audit_fields(tmp_path: Path):
    source, out = tmp_path / "source.tsv", tmp_path / "out.tsv"
    write_tsv(source)
    report = upgrade("witnessed", source, out)
    with out.open(newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["source_audio_review_status"] == ""
    assert row["speaker_identity_basis"] == ""
    assert report["prior_judgments_present"] is False


def test_commentary_blind_upgrade_adds_audio_and_event_modalities(tmp_path: Path):
    source, out = tmp_path / "source.tsv", tmp_path / "out.tsv"
    write_tsv(source)
    report = upgrade("commentary_blind", source, out)
    with out.open(newline="") as handle:
        row = next(csv.DictReader(handle, delimiter="\t"))
    assert row["source_audio_review_status"] == ""
    assert row["event_evidence_modalities"] == ""
    assert report["prior_judgments_present"] is False
