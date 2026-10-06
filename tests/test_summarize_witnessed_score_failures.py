import pytest

from scripts.summarize_witnessed_score_failures import error_class, summarize


def test_successful_retry_wins_and_outstanding_errors_are_latest_only():
    manifest = [{"candidate_id": "a"}, {"candidate_id": "b"}, {"candidate_id": "c"}]
    outputs = [
        {"candidate_id": "a", "model": "q", "error": "timeout", "result": None},
        {"candidate_id": "a", "model": "q", "error": None, "result": {}},
        {"candidate_id": "b", "model": "q", "error": "old", "result": None},
        {"candidate_id": "b", "model": "q", "error": "parse", "result": None},
    ]
    report = summarize(manifest, outputs, "q")
    assert report["successful_candidates"] == 1
    assert report["outstanding_candidates"] == 2
    assert report["latest_error_counts"] == {"missing_result": 1, "parse": 1}
    assert report["attempt_count_histogram"] == {"0": 1, "2": 2}


def test_manifest_identity_contract_fails_closed():
    with pytest.raises(ValueError, match="duplicate"):
        summarize([{"candidate_id": "a"}, {"candidate_id": "a"}], [], "q")


def test_error_class_groups_permission_failures_without_path_cardinality():
    assert error_class(
        "PermissionError: [Errno 13] Permission denied: '/afs/a/media/one.mp4'"
    ) == "PermissionError: [Errno 13] Permission denied: '<path>'"
