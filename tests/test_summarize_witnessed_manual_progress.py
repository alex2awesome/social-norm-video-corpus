import pytest

from scripts.summarize_witnessed_manual_progress import summarize


def selection(*candidate_ids):
    return [{
        "item_id": "clip-a",
        "uid": "youtube__a",
        "candidates": [{"candidate_id": value} for value in candidate_ids],
    }]


def row(candidate_id, *, completed=True):
    value = {
        "candidate_id": candidate_id,
        "item_id": "clip-a",
        "uid": "youtube__a",
        "reaction_grounded": "yes",
        "action_before_or_overlaps_response": "yes",
        "response_targets_action": "yes",
        "responder_role": "separate_bystander",
        "response_content": "targeted_objection",
        "trigger_kind": "interpersonal_treatment",
        "staging": "no_clear_staging_evidence",
        "manual_evidence": "A third party objects to the visible action.",
    }
    if not completed:
        for field in (
            "reaction_grounded", "action_before_or_overlaps_response",
            "response_targets_action", "responder_role", "response_content",
            "trigger_kind", "staging", "manual_evidence",
        ):
            value[field] = ""
    return value


def test_partial_ledger_reports_progress_without_authorizing_metrics():
    report = summarize(selection("a", "b"), [row("a"), row("b", completed=False)])
    assert report["completed_candidates"] == 1
    assert report["blank_candidates"] == 1
    assert report["completion_fraction"] == 0.5
    assert report["completed_strict_social_candidates_descriptive_only"] == 1
    assert report["manual_review_complete"] is False
    assert report["final_metrics_authorized"] is False


def test_complete_ledger_authorizes_final_evaluator_but_not_metrics_itself():
    report = summarize(selection("a"), [row("a")])
    assert report["manual_review_complete"] is True
    assert report["final_metrics_authorized"] is True


def test_partially_filled_candidate_is_rejected():
    partial = row("a", completed=False)
    partial["reaction_grounded"] = "yes"
    with pytest.raises(ValueError, match="partially completed"):
        summarize(selection("a"), [partial])


def test_lineage_and_exact_candidate_coverage_are_required():
    wrong = row("a")
    wrong["uid"] = "youtube__wrong"
    with pytest.raises(ValueError, match="lineage mismatch"):
        summarize(selection("a"), [wrong])
    with pytest.raises(ValueError, match="exactly cover"):
        summarize(selection("a", "b"), [row("a")])


def test_media_failure_abstention_is_counted_explicitly():
    failure = row("a")
    failure.update({
        "reaction_grounded": "uncertain",
        "action_before_or_overlaps_response": "uncertain",
        "response_targets_action": "uncertain",
        "responder_role": "offscreen_or_unresolved",
        "response_content": "uncertain",
        "trigger_kind": "uncertain",
        "staging": "uncertain",
        "manual_evidence": "No playable media was available.",
    })
    report = summarize(selection("a"), [failure])
    assert report["media_failure_abstentions"] == 1
    assert report["completed_strict_social_candidates_descriptive_only"] == 0
