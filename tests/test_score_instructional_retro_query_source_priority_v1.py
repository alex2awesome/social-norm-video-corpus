import json

from scripts.score_instructional_retro_query_source_priority_v1 import run, score_metadata


def test_score_metadata_prioritizes_only_literal_retro_source(tmp_path) -> None:
    source = tmp_path / "dailymotion__x"
    source.mkdir()
    metadata = source / "metadata.json"
    metadata.write_text(json.dumps({
        "provenance": {"query_source": "retro_instr_scan"},
        "demos": [{"clip": "demo_0.mp4"}, {"clip": "demo_1.mp4"}],
    }))
    rows = score_metadata(metadata)
    assert [row["triggered"] for row in rows] == [True, True]
    assert all(row["review_priority"] == "high_review" for row in rows)
    assert all(row["automatic_acceptance"] is False for row in rows)
    assert all(row["delete_media"] is False for row in rows)


def test_nonretro_source_is_preserved_at_standard_priority(tmp_path) -> None:
    source = tmp_path / "dailymotion__y"
    source.mkdir()
    metadata = source / "metadata.json"
    original = json.dumps({
        "provenance": {"query_source": "instructional"},
        "demos": [{"clip": "demo_0.mp4"}],
    })
    metadata.write_text(original)
    rows = score_metadata(metadata)
    assert rows[0]["triggered"] is False
    assert rows[0]["review_priority"] == "standard"
    assert metadata.read_text() == original


def test_run_scores_every_demo_without_mutating_sources(tmp_path) -> None:
    root = tmp_path / "instructional"
    for uid, query_source, demos in (
        ("a", "retro_instr_scan", 2),
        ("b", "instructional", 1),
    ):
        source = root / uid
        source.mkdir(parents=True)
        (source / "metadata.json").write_text(json.dumps({
            "provenance": {"query_source": query_source},
            "demos": [{"clip": f"demo_{i}.mp4"} for i in range(demos)],
        }))
    output = tmp_path / "scores.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = run(root, output, summary_path)
    assert summary["sources_scored"] == 2
    assert summary["demo_clips_scored"] == 3
    assert summary["priority_sources"] == 1
    assert summary["priority_demo_clips"] == 2
    assert len(output.read_text().splitlines()) == 3
    assert summary["corpus_mutated"] is False
