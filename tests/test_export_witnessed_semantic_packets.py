import json
import sqlite3
from pathlib import Path

import pytest

from scripts.export_witnessed_semantic_packets import export_packets, parse_item_id


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "root"
    uid = "dailymotion__abc"
    write_json(
        root / "data" / "hits" / uid / "metadata.json",
        {
            "provenance": {
                "category": "queue_cutting",
                "modality": "witnessed",
                "agent": "human",
                "found_by_query": "queue argument",
                "query_source": "taxonomy",
            },
            "reactions": [
                {
                    "clip_idx": 0,
                    "clip_window": [10.0, 20.0],
                    "phrase": "that is not okay",
                    "matched_text": "not okay",
                    "start": 18.0,
                    "end": 18.5,
                    "context": "you cannot do that",
                    "tier": 1,
                }
            ],
        },
    )
    write_json(
        root / "data" / "transcripts" / f"{uid}.json",
        {
            "segments": [
                {"start": 6.0, "end": 8.0, "text": "outside"},
                {"start": 8.0, "end": 12.0, "text": "first"},
                {"start": 18.0, "end": 21.0, "text": "second"},
                {"start": 24.0, "end": 25.0, "text": "outside"},
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
        (uid, "A title", "channel", "url", 30.0, "dailymotion", "done"),
    )
    connection.commit()
    connection.close()
    selection = tmp_path / "selection.jsonl"
    write_jsonl(
        selection,
        [
            {
                "item_id": f"witnessed:{uid}:0",
                "audit_index": 4,
                "norm": "fairness",
                "found_by_query": "query",
                "query_source": "taxonomy",
                "score_band": "middle",
                "activity_percentile": 0.5,
            }
        ],
    )
    return root, selection


def test_export_packet_resolves_reaction_title_and_excerpt(tmp_path: Path) -> None:
    root, selection = build_fixture(tmp_path)
    packets = export_packets(root, selection)
    assert len(packets) == 1
    packet = packets[0]
    assert packet["title_record"]["title"] == "A title"
    assert packet["reaction"]["matched_text"] == "not okay"
    assert packet["clip_window"] == [10.0, 20.0]
    # The exporter intentionally includes any segment overlapping the
    # three-second context padding around the clip window.
    assert packet["transcript_excerpt"] == "outside first second"


def test_export_uses_item_suffix_as_reaction_ordinal(tmp_path: Path) -> None:
    root, selection = build_fixture(tmp_path)
    metadata_path = root / "data" / "hits" / "dailymotion__abc" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    second = dict(metadata["reactions"][0])
    second["matched_text"] = "second reaction"
    metadata["reactions"].append(second)
    write_json(metadata_path, metadata)
    rows = [json.loads(line) for line in selection.read_text().splitlines()]
    rows[0]["item_id"] = "witnessed:dailymotion__abc:1"
    write_jsonl(selection, rows)
    packet = export_packets(root, selection)[0]
    assert packet["reaction_ordinal"] == 1
    assert packet["clip_idx"] == 0
    assert packet["reaction"]["matched_text"] == "second reaction"
    assert len(packet["clip_reactions"]) == 2


def test_export_fails_if_reaction_ordinal_is_out_of_range(tmp_path: Path) -> None:
    root, selection = build_fixture(tmp_path)
    rows = [json.loads(line) for line in selection.read_text().splitlines()]
    rows[0]["item_id"] = "witnessed:dailymotion__abc:9"
    write_jsonl(selection, rows)
    with pytest.raises(ValueError, match="out of range"):
        export_packets(root, selection)


@pytest.mark.parametrize("item_id", ["bad", "instructional:uid:0", "witnessed:uid:x"])
def test_item_id_must_be_witnessed_and_indexed(item_id: str) -> None:
    with pytest.raises(ValueError):
        parse_item_id(item_id)
