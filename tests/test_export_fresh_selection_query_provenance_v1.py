import json
from pathlib import Path

from scripts.export_fresh_selection_query_provenance_v1 import export


def test_export_preserves_conflicts_and_prefers_selection(tmp_path: Path):
    metadata = tmp_path / "data/instructional/u/metadata.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(json.dumps({
        "provenance": {
            "found_by_query": "metadata query",
            "query_source": "instructional",
            "category": "instr_family",
        }
    }))
    rows, summary = export(
        tmp_path,
        [{
            "item_id": "instructional:u:0", "uid": "u",
            "found_by_query": "selection query", "source_platform": "youtube",
        }],
        [],
        {"u": {"query": "db query", "query_source": "taxonomy"}},
    )
    assert rows[0]["query"] == "selection query"
    assert rows[0]["query_conflict"] is True
    assert rows[0]["query_source"] == "instructional"
    assert rows[0]["query_source_conflict"] is True
    assert summary["by_pillar"] == {"instructional": 1, "witnessed": 0}


def test_export_uses_database_when_metadata_missing(tmp_path: Path):
    rows, summary = export(
        tmp_path,
        [],
        [{"item_id": "witnessed:w:clip_0", "uid": "w", "cohort": "uniform"}],
        {"w": {"query": "bystander intervenes", "query_source": "taxonomy"}},
    )
    assert rows[0]["metadata_found"] is False
    assert rows[0]["db_row_found"] is True
    assert rows[0]["query"] == "bystander intervenes"
    assert summary["items"] == 1
