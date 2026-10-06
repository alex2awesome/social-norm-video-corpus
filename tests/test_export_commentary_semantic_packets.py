import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.export_commentary_semantic_packets import export_packets, parse_item_id


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = tmp_path / "root"
    uid = "dailymotion__comment"
    write_json(
        root / "data" / "discussion" / f"{uid}.json",
        {
            "title": "Example",
            "source": "dailymotion",
            "category": "theft_property",
            "modality": "commentary",
            "provenance": {
                "found_by_query": "package theft",
                "query_source": "taxonomy",
            },
            "statements": [
                {
                    "quote": "That was wrong",
                    "norm": "do not steal",
                    "signal": "disapproval",
                    "start": 12.0,
                    "end": 13.0,
                }
            ],
        },
    )
    write_json(
        root / "data" / "transcripts" / f"{uid}.json",
        {
            "segments": [
                {"start": 9.0, "end": 11.0, "text": "A person takes a box."},
                {"start": 12.0, "end": 13.0, "text": "That was wrong."},
            ]
        },
    )
    db = root / "data" / "state.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db)
    connection.execute(
        "CREATE TABLE seen_videos "
        "(video_id TEXT PRIMARY KEY, title TEXT, channel TEXT, url TEXT, "
        "duration REAL, source TEXT, status TEXT)"
    )
    connection.execute(
        "INSERT INTO seen_videos VALUES (?, ?, ?, ?, ?, ?, ?)",
        (uid, "Example", "channel", "url", 30.0, "dailymotion", "done"),
    )
    connection.commit()
    connection.close()
    item_id = f"commentary:{uid}:0"
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "item_id": item_id,
                "audit_index": 4,
                "norm": "do not steal",
                "stratum": {
                    "polarity": "disapproval",
                    "category": "theft_property",
                },
            }
        )
        + "\n"
    )
    blind = tmp_path / "blind.jsonl"
    blind.write_text(
        json.dumps(
            {
                "item_id": item_id,
                "audit_index": 4,
                "uid": uid,
                "media": {
                    "sample_start_sec": 8.0,
                    "sample_end_sec": 16.0,
                },
            }
        )
        + "\n"
    )
    return root, selection, blind


def test_export_packet(tmp_path: Path) -> None:
    packet = export_packets(*fixture(tmp_path))[0]
    assert packet["statement"]["quote"] == "That was wrong"
    assert packet["visual_window"] == {"start": 8.0, "end": 16.0}
    assert packet["selection"]["polarity"] == "disapproval"
    assert "person takes a box" in packet["transcript_excerpt"]


def test_direct_script_invocation(tmp_path: Path) -> None:
    root, selection, blind = fixture(tmp_path)
    output = tmp_path / "packets.jsonl"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "export_commentary_semantic_packets.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--selection",
            str(selection),
            "--blind-manifest",
            str(blind),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["items"] == 1
    assert json.loads(output.read_text())["statement"]["norm"] == "do not steal"


def test_coverage_mismatch_fails(tmp_path: Path) -> None:
    root, selection, blind = fixture(tmp_path)
    blind.write_text("")
    with pytest.raises(ValueError, match="coverage mismatch"):
        export_packets(root, selection, blind)


def test_out_of_range_statement_fails(tmp_path: Path) -> None:
    root, selection, blind = fixture(tmp_path)
    row = json.loads(selection.read_text())
    row["item_id"] = row["item_id"].rsplit(":", 1)[0] + ":3"
    selection.write_text(json.dumps(row) + "\n")
    blind_row = json.loads(blind.read_text())
    blind_row["item_id"] = row["item_id"]
    blind.write_text(json.dumps(blind_row) + "\n")
    with pytest.raises(ValueError, match="out of range"):
        export_packets(root, selection, blind)


@pytest.mark.parametrize(
    "item_id", ["bad", "witnessed:u:0", "commentary:u:x", "commentary:u:-1"]
)
def test_parse_item_id_fails_closed(item_id: str) -> None:
    with pytest.raises(ValueError):
        parse_item_id(item_id)
