import json

from scripts.score_instructional_combined_review_tiers_v1 import score_corpus, score_metadata


def write_source(tmp_path, uid, query_source, polarities):
    source = tmp_path / uid
    source.mkdir(parents=True)
    (source / "metadata.json").write_text(json.dumps({
        "provenance": {"query_source": query_source},
        "demos": [{"polarity": value, "clip": f"{index}.mp4"} for index, value in enumerate(polarities)],
    }))
    return source / "metadata.json"


def test_all_four_combinations_are_explicit_and_non_destructive(tmp_path):
    retro = score_metadata(write_source(tmp_path, "a", "retro_instr_scan", ["violation", "explanation"]))
    ordinary = score_metadata(write_source(tmp_path, "b", "instructional", ["correct", "explanation"]))
    assert [row["combined_review_tier"] for row in retro + ordinary] == [
        "tier_1_both",
        "tier_1_retro_out_of_combination_support",
        "tier_2_polarity_only",
        "tier_3_lower_priority_preserved",
    ]
    assert all(row["manual_review_required"] for row in retro + ordinary)
    assert not any(row["automatic_acceptance"] for row in retro + ordinary)
    assert not any(row["automatic_rejection"] for row in retro + ordinary)
    assert not any(row["delete_media"] for row in retro + ordinary)


def test_bad_metadata_is_preserved_as_unscored(tmp_path):
    source = tmp_path / "bad"
    source.mkdir()
    (source / "metadata.json").write_text("not json")
    rows = score_corpus(tmp_path)
    assert len(rows) == 1
    assert rows[0]["combined_review_tier"] == "unscored_preserved"
    assert rows[0]["error"].startswith("JSONDecodeError")
    assert rows[0]["corpus_disposition"] is None
