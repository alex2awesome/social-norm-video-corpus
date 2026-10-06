import json

from scripts.score_instructional_scene_title_cues import run, score_metadata


def test_score_metadata_is_review_only(tmp_path) -> None:
    source = tmp_path / "youtube__x"
    source.mkdir()
    path = source / "metadata.json"
    path.write_text(
        json.dumps(
            {
                "title": "Animated Social Story About Sharing",
                "demos": [{}, {}],
                "provenance": {"found_by_query": "sharing story"},
            }
        )
    )
    row = score_metadata(path)
    assert row["scene_title_candidate"] is True
    assert row["demo_count"] == 2
    assert row["automatic_acceptance"] is False


def test_run_scores_all_sources_without_mutating_metadata(tmp_path) -> None:
    root = tmp_path / "instructional"
    source = root / "dailymotion__x"
    source.mkdir(parents=True)
    metadata = source / "metadata.json"
    original = json.dumps({"title": "Interview About a Social Story", "demos": [{}]})
    metadata.write_text(original)
    output = tmp_path / "scores.jsonl"
    summary_path = tmp_path / "summary.json"
    summary = run(root, output, summary_path)
    assert summary["sources_scored"] == 1
    assert summary["sources_selected_for_priority_review"] == 0
    assert metadata.read_text() == original
