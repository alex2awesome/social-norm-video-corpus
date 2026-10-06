from scripts.validate_witnessed_video_asr_population_lineage import summarize


def parent(item_id="witnessed:youtube__a:0", uid="youtube__a"):
    return {
        "item_id": item_id,
        "uid": uid,
        "candidates": [
            {"segment_index": 1, "text": "Stop that."},
            {"segment_index": 2, "text": "I did nothing.", "negative_self_defense": True},
            {"segment_index": 3, "text": "She said stop.", "negative_reported": True},
        ],
    }


def rendered(candidate_id="witnessed:youtube__a:0:candidate_1"):
    return {
        "candidate_id": candidate_id,
        "item_id": "witnessed:youtube__a:0",
        "uid": "youtube__a",
        "source_item_id": "witnessed:youtube__a:source",
        "candidate_text": "Stop that.",
    }


def test_lineage_accounts_for_explicit_negative_exclusions():
    report = summarize(
        [parent()], [rendered()], expected_proposals=3,
        expected_proposal_parents=1, expected_render_source_items=1,
    )
    assert report["passed"] is True
    assert report["proposed_candidate_windows"] == 3
    assert report["excluded_candidate_windows"] == 2
    assert report["eligible_candidate_windows"] == 1
    assert report["coverage_fraction"] == 1


def test_lineage_fails_closed_on_missing_and_unexpected_rows():
    report = summarize(
        [parent()], [rendered("unexpected")], expected_proposals=3,
        expected_proposal_parents=1, expected_render_source_items=1,
    )
    kinds = {issue["kind"] for issue in report["issues"]}
    assert report["passed"] is False
    assert "eligible_candidate_missing_from_manifest" in kinds
    assert "unexpected_manifest_candidate" in kinds


def test_lineage_detects_field_mismatch():
    wrong = rendered()
    wrong["candidate_text"] = "Different words"
    report = summarize([parent()], [wrong])
    assert report["passed"] is False
    assert report["issues"][0]["kind"] == "proposal_manifest_field_mismatch"


def test_lineage_distinguishes_proposal_parents_from_render_source_items():
    second = parent("witnessed:youtube__a:1")
    second["candidates"] = [{"segment_index": 4, "text": "Please stop."}]
    second_render = {
        **rendered("witnessed:youtube__a:1:candidate_4"),
        "item_id": "witnessed:youtube__a:1",
        "candidate_text": "Please stop.",
    }
    report = summarize(
        [parent(), second], [rendered(), second_render],
        expected_proposals=4, expected_proposal_parents=2,
        expected_render_source_items=1,
    )
    assert report["passed"] is True
    assert report["proposal_parent_rows"] == 2
    assert report["unique_render_source_items"] == 1
