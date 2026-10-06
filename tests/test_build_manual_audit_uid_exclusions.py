import json

from scripts.build_manual_audit_uid_exclusions import (
    collect_from_file,
    json_artifacts,
    manual_artifacts,
)


def test_collects_uids_and_normalizes_item_ids(tmp_path):
    manual = tmp_path / "manual_review.jsonl"
    manual.write_text(
        "\n".join(
            [
                json.dumps({"uid": "source-a"}),
                json.dumps(
                    {
                        "nested": [
                            {"item_id": "instructional:source-b:2"},
                            {"candidate_id": "commentary:source-c:0"},
                        ]
                    }
                ),
            ]
        )
        + "\n"
    )
    ignored = tmp_path / "model_scores.jsonl"
    ignored.write_text(json.dumps({"uid": "not-manual"}) + "\n")

    result = set()
    assert collect_from_file(manual, result)
    assert result == {"source-a", "source-b", "source-c"}
    assert manual_artifacts([tmp_path]) == [manual]
    assert json_artifacts([tmp_path]) == [manual, ignored]


def test_collects_candidate_ids_from_manual_tsv(tmp_path):
    ledger = tmp_path / "manual_atomic_ledger.tsv"
    ledger.write_text(
        "candidate_id\tmanual_note\n"
        "witnessed:youtube__a:0:candidate_1\tchecked\n"
        "witnessed:dailymotion__b:2:candidate_3\tchecked\n"
    )
    result = set()
    assert collect_from_file(ledger, result)
    assert result == {"youtube__a", "dailymotion__b"}
    assert ledger in manual_artifacts([tmp_path])
