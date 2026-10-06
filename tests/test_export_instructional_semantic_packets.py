import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.export_instructional_semantic_packets import export_packets, parse_item_id


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "root"
    uid = "dailymotion__demo"
    write_json(
        root / "data" / "instructional" / uid / "metadata.json",
        {
            "genre": "training",
            "provenance": {"category": "instr_customer_service"},
            "demos": [
                {
                    "start": 10.0,
                    "end": 20.0,
                    "clip": "demo_0.mp4",
                    "polarity": "violation",
                    "norm": "be polite",
                    "start_quote": "start",
                    "end_quote": "end",
                    "explanation": "A rude interaction.",
                }
            ],
        },
    )
    write_json(
        root / "data" / "transcripts" / f"{uid}.json",
        {"segments": [{"start": 9.0, "end": 12.0, "text": "start text"}]},
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
        (uid, "Demo title", "channel", "url", 30.0, "dailymotion", "done"),
    )
    connection.commit()
    connection.close()
    selection = tmp_path / "selection.jsonl"
    selection.write_text(
        json.dumps(
            {
                "item_id": f"instructional:{uid}:0",
                "audit_index": 2,
                "norm": "be polite",
                "stratum": {
                    "polarity": "violation",
                    "category": "instr_customer_service",
                },
            }
        )
        + "\n"
    )
    return root, selection


def test_export_packet(tmp_path: Path) -> None:
    packet = export_packets(*fixture(tmp_path))[0]
    assert packet["title_record"]["title"] == "Demo title"
    assert packet["demo"]["norm"] == "be polite"
    assert packet["selection"]["polarity"] == "violation"
    assert packet["transcript_excerpt"] == "start text"


def test_direct_script_invocation(tmp_path: Path) -> None:
    root, selection = fixture(tmp_path)
    output = tmp_path / "packets.jsonl"
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "export_instructional_semantic_packets.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--selection",
            str(selection),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["items"] == 1
    assert json.loads(output.read_text())["item_id"] == "instructional:dailymotion__demo:0"


def test_out_of_range_demo_fails(tmp_path: Path) -> None:
    root, selection = fixture(tmp_path)
    row = json.loads(selection.read_text())
    row["item_id"] = "instructional:dailymotion__demo:3"
    selection.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="out of range"):
        export_packets(root, selection)


@pytest.mark.parametrize(
    "item_id", ["bad", "witnessed:u:0", "instructional:u:x", "instructional:u:-1"]
)
def test_parse_item_id_fails_closed(item_id: str) -> None:
    with pytest.raises(ValueError):
        parse_item_id(item_id)
