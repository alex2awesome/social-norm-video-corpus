import hashlib

import pytest

from scripts.seal_post_localization_source_lineage_v1 import seal


def row(path):
    return {
        "candidate_id": "c", "source_path": str(path),
        "approval_status": "unreviewed_localization_candidate",
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }


def test_seals_unapproved_source_without_approving_or_mutating(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    value = seal([row(source)])[0]
    assert value["source_sha256"] == hashlib.sha256(b"source").hexdigest()
    assert value["source_lineage_sealed"] is True
    assert value["automatic_acceptance"] is False
    assert value["corpus_mutation_authorized"] is False


def test_rejects_changed_previously_sealed_source(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    value = row(source)
    value["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="hash mismatch"):
        seal([value])
