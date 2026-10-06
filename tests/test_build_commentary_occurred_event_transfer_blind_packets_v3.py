import json

import pytest

from scripts.build_commentary_occurred_event_transfer_blind_packets_v3 import FORBIDDEN, build_packets
from scripts.select_commentary_occurred_event_transfer_v3 import sha256


def test_builds_transcript_only_packet_without_label_leakage(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": [{
        "item_id": "commentary:u:0",
        "uid": "u",
        "title": "leaky title",
        "norm": "theft",
        "start_sec": 1,
        "end_sec": 2,
        "transcript_context": [{"start": 0, "end": 3, "text": "A person took a wallet."}],
    }]}))
    selected = [{
        "transfer_index": 0,
        "item_id": "commentary:u:0",
        "source_manifest_path": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "content_manifest_sha256": "content",
    }]
    packets = build_packets(selected)
    assert len(packets) == 1
    assert not (set(packets[0]) & FORBIDDEN)
    assert packets[0]["transcript_context"][0]["text"] == "A person took a wallet."


def test_rejects_changed_source_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"items": []}))
    with pytest.raises(ValueError, match="hash"):
        build_packets([{
            "transfer_index": 0,
            "item_id": "x",
            "source_manifest_path": str(manifest),
            "source_manifest_sha256": "bad",
        }])
