from pathlib import Path

import pytest

from scripts.render_commentary_motion_proxies import proxy_command, select_unresolved


def test_select_unresolved_joins_only_u_rows() -> None:
    semantic = [
        {"candidate_id": "c0", "audit_index": 0, "norm": "n0"},
        {"candidate_id": "c1", "audit_index": 1, "norm": "n1"},
    ]
    ledger = [
        {
            "candidate_id": "c0",
            "visual_status": "Y",
            "literal_description": "clear",
            "failure_or_followup": "",
        },
        {
            "candidate_id": "c1",
            "visual_status": "U",
            "literal_description": "unclear",
            "failure_or_followup": "motion",
        },
    ]
    selected = select_unresolved(semantic, ledger)
    assert [row["candidate_id"] for row in selected] == ["c1"]
    assert selected[0]["blind_visual_status"] == "U"
    assert selected[0]["norm"] == "n1"


def test_select_unresolved_rejects_missing_semantics() -> None:
    with pytest.raises(ValueError, match="absent"):
        select_unresolved(
            [],
            [
                {
                    "candidate_id": "missing",
                    "visual_status": "U",
                    "literal_description": "",
                    "failure_or_followup": "",
                }
            ],
        )


def test_proxy_command_preserves_optional_audio_and_caps_resolution() -> None:
    command = proxy_command(
        Path("/env/bin/ffmpeg"),
        Path("/data/source.mp4"),
        Path("/audit/proxy.mp4"),
    )
    assert command[0] == "/env/bin/ffmpeg"
    assert "0:a:0?" in command
    assert "scale=w='min(360,iw)':h=-2,fps=8" in command
    assert command[-1] == "/audit/proxy.mp4"
