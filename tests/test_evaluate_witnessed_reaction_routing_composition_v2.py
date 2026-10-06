import pytest

from scripts.evaluate_witnessed_reaction_routing_composition_v2 import (
    evaluate,
    source_exclusion,
)


def result(
    candidate_id="c", item_id="i", uid="dailymotion__u", *,
    staging="no_clear_staging_evidence", role="separate_bystander",
):
    return {
        "candidate_id": candidate_id,
        "item_id": item_id,
        "uid": uid,
        "reaction_grounded": "yes",
        "action_before_or_overlaps_response": "yes",
        "response_targets_action": "yes",
        "responder_role": role,
        "response_content": "targeted_objection",
        "trigger_kind": "interpersonal_treatment",
        "staging": staging,
        "manual_evidence": "A separate person directly objects to the action.",
    }


def selection(candidate_id="c", item_id="i", uid="dailymotion__u"):
    return [{
        "item_id": item_id,
        "uid": uid,
        "cohort": "uniform_probability_sample",
        "candidates": [{"candidate_id": candidate_id}],
    }]


def model(row, *, error=None):
    return [{
        "candidate_id": row["candidate_id"],
        "item_id": row["item_id"],
        "uid": row["uid"],
        "error": error,
        "result": None if error else {
            key: row[key] for key in (
                "reaction_grounded", "action_before_or_overlaps_response",
                "response_targets_action", "responder_role", "response_content",
                "trigger_kind", "staging",
            )
        },
    }]


def provenance(
    item_id="i", uid="dailymotion__u", *, title="ordinary clip", channel="x",
    title_staging=False, creator=False,
):
    return [{
        "item_id": item_id,
        "uid": uid,
        "cohort": "uniform_probability_sample",
        "title": title,
        "channel": channel,
        "title_staging_cue": title_staging,
        "title_creator_initiated_candidate_cue": creator,
    }]


def source_review(
    item_id="i", uid="dailymotion__u", *,
    mechanisms="witnessed_creator_staging_title_v2",
):
    return [{
        "item_id": item_id,
        "uid": uid,
        "cohort": "uniform_probability_sample",
        "triggered_mechanisms": mechanisms,
        "strict_organic_exclusion_supported": "yes",
        "instructional_demo_accepted": "not_reviewed",
        "manual_rationale": "The source title explicitly identifies a produced prank.",
    }]


def routes(report):
    return report["clip_cohort_metrics"]["uniform_probability_sample"]["routes"]


def test_staged_scene_remains_a_reaction_positive_but_not_organic():
    row = result(staging="clearly_staged")
    report = evaluate(selection(), [row], model(row), provenance(), [])
    values = routes(report)
    reaction = values["strict_bystander_reaction_staging_independent"]["fail_closed"]
    organic = values["organic_reaction_vlm_staging_only"]["fail_closed"]
    assert reaction["tp"] == 1
    assert organic["selected"] == 0
    assert organic["tn"] == 1


def test_audited_title_cue_reroutes_without_erasing_reaction():
    row = result()
    prov = provenance(
        title="A public phone prank (social experiment)", title_staging=True
    )
    report = evaluate(selection(), [row], model(row), prov, source_review())
    values = routes(report)
    assert values["strict_bystander_reaction_staging_independent"]["fail_closed"]["tp"] == 1
    routed = values["organic_reaction_audited_source_cues_only"]["fail_closed"]
    assert routed["selected"] == 0
    assert routed["tn"] == 1
    audit = report["clip_cohort_metrics"]["uniform_probability_sample"]["audited_source_routing"]
    assert audit["manual_reaction_positive_reroutes"] == ["i"]
    assert audit["blind_manual_organic_labels_overridden_by_source_evidence"] == ["i"]


def test_official_wwyd_channel_is_an_audited_source_exclusion():
    row = {
        "item_id": "i", "title": "A scenario", "channel": "What Would You Do?",
        "title_staging_cue": False,
        "title_creator_initiated_candidate_cue": False,
    }
    excluded, mechanisms = source_exclusion(row)
    assert excluded is True
    assert mechanisms == ("witnessed_official_wwyd_channel_v1",)


def test_source_cue_absence_does_not_create_a_reaction_positive():
    row = result(role="affected_target")
    report = evaluate(selection(), [row], model(row), provenance(), [])
    values = routes(report)
    assert values["strict_bystander_reaction_staging_independent"]["fail_closed"]["tn"] == 1
    assert report["source_cue_absence_certifies_organic"] is False


def test_model_error_fails_closed_for_reaction_recall():
    row = result()
    report = evaluate(
        selection(), [row], model(row, error="timeout"), provenance(), []
    )
    reaction = routes(report)["strict_bystander_reaction_staging_independent"]["fail_closed"]
    assert reaction["fn"] == 1
    assert report["model_candidate_coverage"] == 0


def test_frozen_provenance_must_match_audited_title_scorer():
    row = result()
    with pytest.raises(ValueError, match="disagrees"):
        evaluate(
            selection(), [row], model(row),
            provenance(title="obvious prank", title_staging=False), [],
        )


def test_provenance_must_exactly_cover_selection():
    row = result()
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate(selection(), [row], model(row), [], [])


def test_manual_source_review_must_cover_every_trigger():
    row = result()
    prov = provenance(title="a prank", title_staging=True)
    with pytest.raises(ValueError, match="exactly cover"):
        evaluate(selection(), [row], model(row), prov, [])
