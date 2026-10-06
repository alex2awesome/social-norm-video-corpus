from __future__ import annotations

import sqlite3

import pytest

from scripts.export_witnessed_selection_provenance import export, fetch_rows


def test_fetch_rows_queries_only_selected_ids_and_adapts_schema(tmp_path):
    db = tmp_path / "state.db"
    connection = sqlite3.connect(db)
    connection.execute(
        "CREATE TABLE seen_videos (video_id TEXT PRIMARY KEY, title TEXT, query TEXT, query_source TEXT)"
    )
    connection.executemany(
        "INSERT INTO seen_videos VALUES (?,?,?,?)",
        [("a", "A", "qa", "taxonomy"), ("b", "B", "qb", "related")],
    )
    connection.commit()
    connection.close()
    rows = fetch_rows(db, {"b"})
    assert set(rows) == {"b"}
    assert rows["b"]["query_source"] == "related"


def test_export_preserves_missing_lineage_as_explicit_flags():
    selected = [{"item_id": "witnessed:a:clip_0", "uid": "a", "cohort": "uniform_probability_sample", "platform": "reddit"}]
    rows, summary = export(selected, {}, [])
    assert rows[0]["db_row_found"] is False
    assert rows[0]["query"] is None
    assert summary["db_rows_found"] == 0
    assert summary["corpus_mutated"] is False


def test_export_joins_staging_without_turning_absence_into_acceptance():
    selected = [{"item_id": "witnessed:a:clip_0", "uid": "a", "cohort": "uniform_probability_sample"}]
    rows, summary = export(
        selected,
        {"a": {"video_id": "a", "title": "Prank", "query": "q"}},
        [{"uid": "a", "title_staging_cue": True, "error": None}],
    )
    assert rows[0]["title_staging_cue"] is True
    assert "strict_organic_witnessed_eligible" not in rows[0]
    assert summary["staging_rows_found"] == 1


def test_selection_must_be_source_disjoint():
    selected = [
        {"item_id": "a:0", "uid": "a", "cohort": "x"},
        {"item_id": "a:1", "uid": "a", "cohort": "x"},
    ]
    with pytest.raises(ValueError, match="source-disjoint"):
        export(selected, {}, [])

