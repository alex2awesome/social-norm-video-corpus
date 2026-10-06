import pytest

from scripts.evaluate_commentary_tail_trim_experiment import evaluate_fixed_tail
from tests.test_evaluate_commentary_clip_plan import blind, post


def manifest():
    return [{
        "candidate_id": "a", "parent_candidate_id": "parent-a",
        "audit_index": 0, "variant": "last_half_v1", "proxy_clip": "/a.mp4",
        "proxy_clip_sha256": "hash",
    }]


def prereg(expected=1, fraction=0.5, threshold=0.5):
    return {
        "cohort": {"parents": expected},
        "transform": {
            "name": "last_half_v1" if fraction == 0.5 else "other",
            "rule": "retain exactly the final 50 percent",
        },
        "manual_gate": {
            "minimum_parent_salvage_rate_for_independent_replication": threshold
        },
    }


def test_full_strict_pass_only_makes_experiment_replication_eligible():
    report, accepted = evaluate_fixed_tail(manifest(), blind(), post(), prereg())
    assert len(accepted) == 1
    assert report["parent_salvage_rate"] == 1.0
    assert report["replication_eligible"] is True
    assert report["shadow_transform_candidate_promoted"] is False
    assert report["automatic_acceptance_rule_promoted"] is False
    assert report["corpus_disposition"] is None


def test_label_leak_fails_closed_and_blocks_replication():
    report, accepted = evaluate_fixed_tail(
        manifest(), blind(label_bearing_text_absent="no"), post(), prereg()
    )
    assert accepted == []
    assert report["parent_salvage_rate"] == 0.0
    assert report["replication_eligible"] is False


@pytest.mark.parametrize(
    "changed,match",
    [
        ({"expected": 2}, "preregistered cohort"),
        ({"fraction": 0.4}, "fixed last-half"),
    ],
)
def test_preregistered_cohort_and_transform_fail_closed(changed, match):
    with pytest.raises(ValueError, match=match):
        evaluate_fixed_tail(
            manifest(), blind(), post(),
            prereg(expected=changed.get("expected", 1), fraction=changed.get("fraction", 0.5)),
        )


def test_duplicate_parent_is_rejected():
    rows = manifest() * 2
    rows[1] = dict(rows[1], candidate_id="b", audit_index=1, proxy_clip="/b.mp4")
    with pytest.raises(ValueError, match="unique non-empty parent"):
        evaluate_fixed_tail(rows, blind() + [dict(blind()[0], candidate_id="b")], post() + [dict(post()[0], candidate_id="b")], prereg(expected=2))
