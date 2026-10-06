import pytest

from scripts.run_witnessed_video_asr_postscore_audit import audit_ready, completion


def manifest():
    return [{"candidate_id": "a"}, {"candidate_id": "b"}]


def test_completion_requires_successful_result_for_every_candidate():
    report = completion(
        manifest(),
        [
            {"candidate_id": "a", "result": None, "error": "timeout"},
            {"candidate_id": "a", "result": {"ok": True}, "error": None},
            {"candidate_id": "b", "result": {"ok": True}, "error": None},
        ],
        2,
    )
    assert report["ready"] is True
    assert report["score_rows"] == 3
    assert report["successful_candidate_ids"] == 2


def test_completion_waits_on_error_only_candidate():
    report = completion(
        manifest(), [{"candidate_id": "a", "result": {}, "error": None}], 2
    )
    assert report["ready"] is False
    assert report["remaining_candidate_ids"] == 1


def test_completion_fails_closed_on_unexpected_score_id():
    report = completion(
        manifest(),
        [
            {"candidate_id": "a", "result": {}, "error": None},
            {"candidate_id": "b", "result": {}, "error": None},
            {"candidate_id": "outside", "result": {}, "error": None},
        ],
        2,
    )
    assert report["ready"] is False
    assert report["unexpected_candidate_ids"] == ["outside"]


def test_completion_rejects_manifest_count_or_identity_mismatch():
    with pytest.raises(ValueError, match="manifest candidate-ID contract"):
        completion([{"candidate_id": "a"}, {"candidate_id": "a"}], [], 2)


def test_audit_waits_for_scorer_even_when_coverage_threshold_is_met():
    progress = completion(
        manifest(),
        [
            {"candidate_id": "a", "result": {}, "error": None},
            {"candidate_id": "b", "result": {}, "error": None},
        ],
        2,
    )
    assert audit_ready(progress, scoring_alive=True, minimum_coverage=0.98) is False
    assert audit_ready(progress, scoring_alive=False, minimum_coverage=0.98) is True


def test_audit_accepts_finalized_preregistered_partial_coverage():
    rows = [{"candidate_id": str(index)} for index in range(100)]
    scores = [
        {"candidate_id": str(index), "result": {}, "error": None}
        for index in range(98)
    ]
    progress = completion(rows, scores, 100)
    assert audit_ready(progress, scoring_alive=False, minimum_coverage=0.98) is True
    assert audit_ready(progress, scoring_alive=False, minimum_coverage=0.99) is False
