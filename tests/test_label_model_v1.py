import pytest

from scripts.labeling_functions_v1 import make_lf_record
from scripts.lf_matrix_v1 import build_matrices
from scripts.label_model_v1 import (
    FamilyLabelModel,
    calibration_report,
    collapse_families,
    evaluate_predictions,
    family_vote_matrix,
    majority_vote,
    run_experiment,
    score_matrix,
    shadow_band,
    source_disjoint_split,
    strongest_single_lf,
)


def synthetic_matrix(n=200):
    """Deterministic synthetic corpus: items alternate positive/negative.

    ``good`` families track the truth with occasional structured noise;
    ``noisy`` abstains often and errs more.
    """
    records = []
    for i in range(n):
        item = f"src{i // 2}:item{i}"
        truth = 1 if i % 2 == 0 else -1
        # Noise is parity-symmetric so both classes are corrupted equally.
        good_vote = -truth if i % 20 in {0, 1} else truth  # ~90% accurate
        records.append(
            make_lf_record(
                item_id=item, pillar="instructional", target="demonstration_present",
                lf_id="good_a", family="event_action", vote=good_vote,
            )
        )
        records.append(
            make_lf_record(
                item_id=item, pillar="instructional", target="demonstration_present",
                lf_id="good_b", family="visual_depiction",
                vote=-truth if i % 14 in {0, 1} else truth,
            )
        )
        noisy_vote = 0 if i % 3 == 0 else (-truth if i % 8 in {0, 1} else truth)
        records.append(
            make_lf_record(
                item_id=item, pillar="instructional", target="demonstration_present",
                lf_id="noisy", family="semantic_model_judgment", vote=noisy_vote,
                abstain_reason="no evidence" if noisy_vote == 0 else None,
            )
        )
    matrix = build_matrices(records)[("instructional", "demonstration_present")]
    gold = {
        f"src{i // 2}:item{i}": (1 if i % 2 == 0 else -1) for i in range(n)
    }
    return matrix, gold


def test_family_collapse_negative_wins_within_family():
    matrix = build_matrices(
        [
            make_lf_record(
                item_id="a", pillar="witnessed", target="norm_event_supported",
                lf_id="lf1", family="reaction", vote=1,
            ),
            make_lf_record(
                item_id="a", pillar="witnessed", target="norm_event_supported",
                lf_id="lf2", family="reaction", vote=-1,
            ),
            make_lf_record(
                item_id="a", pillar="witnessed", target="norm_event_supported",
                lf_id="lf3", family="social_context", vote=1,
            ),
        ]
    )[("witnessed", "norm_event_supported")]
    collapsed = collapse_families(matrix, "a")
    assert collapsed == {"reaction": -1, "social_context": 1}
    assert majority_vote(collapsed) == 0


def test_em_recovers_family_reliability_ordering():
    matrix, gold = synthetic_matrix()
    votes = family_vote_matrix(matrix)
    model = FamilyLabelModel(sorted(set(matrix["lfs"].values()))).fit(
        list(votes.values())
    )
    params = model.parameters()
    acc = params["family_accuracy"]
    assert acc["event_action"] > 0.8
    assert acc["event_action"] > acc["semantic_model_judgment"]
    # Posteriors separate the classes.
    predictions = {
        item: 1 if model.posterior_positive(v) >= 0.5 else -1
        for item, v in votes.items()
    }
    stats = evaluate_predictions(predictions, gold)
    assert stats["precision"] > 0.85 and stats["recall"] > 0.85


def test_source_disjoint_split_keeps_groups_together():
    items = [f"src{i // 4}:item{i}" for i in range(100)]
    groups = {item: item.split(":")[0] for item in items}
    split = source_disjoint_split(items, groups)
    by_group = {}
    for item, part in split.items():
        by_group.setdefault(groups[item], set()).add(part)
    assert all(len(parts) == 1 for parts in by_group.values())
    assert set(split.values()) == {"train", "calibration", "test"}
    # Deterministic.
    assert split == source_disjoint_split(items, groups)


def test_gates_dominate_posterior():
    votes = {"event_action": 1, "visual_depiction": 1}
    failed = {"eligible": False, "failed_gates": ["media_decodes"], "unknown_gates": []}
    ok = {"eligible": True, "failed_gates": [], "unknown_gates": []}
    unknown = {"eligible": False, "failed_gates": [], "unknown_gates": ["bounds_valid"]}
    assert shadow_band(0.99, votes, failed) == "gate_failed_review"
    assert shadow_band(0.99, votes, unknown) == "insufficient_evidence_abstain"
    assert shadow_band(0.99, votes, ok) == "high_confidence_candidate"
    # Prior saturation cannot certify an item with only negative evidence.
    assert (
        shadow_band(0.99, {"source_format_provenance": -1}, ok)
        == "insufficient_evidence_abstain"
    )
    assert shadow_band(0.05, votes, ok) == "likely_failure_or_reroute"
    assert shadow_band(0.5, votes, ok) == "insufficient_evidence_abstain"
    assert (
        shadow_band(0.99, {"event_action": 1, "reaction": -1}, ok)
        == "disagreement_manual_review"
    )
    assert shadow_band(0.99, {}, ok) == "insufficient_evidence_abstain"


def test_score_matrix_emits_shadow_only_rows():
    matrix, _ = synthetic_matrix(40)
    model, rows = score_matrix(
        matrix, pillar="instructional", target="demonstration_present"
    )
    assert len(rows) == 40
    for row in rows:
        assert row["acceptance_label"] is None
        assert row["corpus_disposition"] is None
        assert row["delete_media"] is False
        assert row["shadow_only"] is True
        assert 0.0 <= row["posterior_positive"] <= 1.0


def test_run_experiment_reports_baselines_and_calibration():
    matrix, gold = synthetic_matrix(300)
    groups = {item: item.split(":")[0] for item in matrix["items"]}
    report = run_experiment(
        matrix, gold, groups, pillar="instructional", target="demonstration_present"
    )
    assert report["split_sizes"]["train"] > 0
    assert report["split_sizes"]["test"] > 0
    assert report["baselines"]["majority_vote"]["precision"] is not None
    assert report["baselines"]["strongest_single_lf"]["lf_id"] in {
        "good_a", "good_b", "noisy",
    }
    assert report["label_model_test"]["precision"] > 0.8
    assert report["policy"] == "shadow_only_no_acceptance_no_deletion"
    assert report["calibration"]["n"] > 0
    assert report["calibration"]["expected_calibration_error"] < 0.5


def test_strongest_single_lf_uses_train_split_only():
    matrix, gold = synthetic_matrix(100)
    train = {item for i, item in enumerate(sorted(matrix["items"])) if i < 50}
    best = strongest_single_lf(matrix, gold, train)
    assert best in {"good_a", "good_b"}


def test_calibration_report_math():
    report = calibration_report([0.9, 0.9, 0.1, 0.1], [1, 1, -1, -1], bins=2)
    assert report["n"] == 4
    assert report["expected_calibration_error"] == pytest.approx(0.1)
    with pytest.raises(ValueError):
        calibration_report([0.5], [1, -1])


def test_fit_rejects_empty_input():
    with pytest.raises(ValueError):
        FamilyLabelModel(["event_action"]).fit([])


def test_em_cannot_polarity_flip_a_minority_negative_family():
    # A dominant correlated positive block must not make EM invert the sign
    # of a sparse negative-voting family (the v2 corpus failure mode).
    rows = []
    for i in range(200):
        row = {"event_action": 1, "social_context": 1}
        if i % 8 == 0:
            row["source_format_provenance"] = -1
        rows.append(row)
    model = FamilyLabelModel(
        ["event_action", "social_context", "source_format_provenance"]
    ).fit(rows)
    for family, accuracy in model.parameters()["family_accuracy"].items():
        assert accuracy >= 0.55, family
    flagged = model.posterior_positive(
        {"event_action": 1, "social_context": 1, "source_format_provenance": -1}
    )
    clean = model.posterior_positive({"event_action": 1, "social_context": 1})
    assert flagged < clean  # a negative vote must still count against
