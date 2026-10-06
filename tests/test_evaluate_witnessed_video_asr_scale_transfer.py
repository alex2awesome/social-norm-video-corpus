from scripts.evaluate_witnessed_video_asr_scale_transfer import evaluate


def fixtures(precision=0.75, recall=0.8):
    prereg = {"required_gate": {
        "minimum_precision": 0.7, "minimum_recall": 0.8,
        "minimum_manual_clips": 24,
    }}
    clip = {
        "manual_clips": 24,
        "rules": {"strict_bystander_reaction": {"fail_closed": {
            "precision": precision, "recall": recall, "tp": 4, "fp": 1,
            "fn": 1, "selected": 5,
        }}},
    }
    manifest = [{
        "candidate_id": "a", "two_stage_full_proxy_then_candidate_cut": True,
        "max_source_frames": 96, "max_width": 512,
    }]
    model = [{"candidate_id": "a", "error": None}]
    changes = [{"change_audit": "response_present:improved"}]
    return prereg, clip, manifest, model, changes


def test_gate_passes_only_for_shadow_ranking_scope():
    report = evaluate(*fixtures())
    assert report["passed"] is True
    assert report["automatic_acceptance"] is False
    assert report["decision_change_audit"]["atom_change_outcomes"] == {"improved": 1}


def test_gate_fails_incomplete_model_coverage():
    prereg, clip, manifest, _model, changes = fixtures()
    report = evaluate(prereg, clip, manifest, [], changes)
    assert report["passed"] is False
    assert report["status"] == "failed_transfer_do_not_scale"


def test_gate_fails_nonmatching_representation_or_recall():
    prereg, clip, manifest, model, changes = fixtures(recall=0.79)
    manifest[0]["max_source_frames"] = 95
    report = evaluate(prereg, clip, manifest, model, changes)
    assert report["passed"] is False
    assert not report["gate_checks"]["representation_valid"]
    assert not report["gate_checks"]["recall_at_least_minimum"]
