import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREREG = ROOT / "audit_runs/20260806_commentary_hierarchical_full_source_v2/preregistration.json"


def test_preregistration_freezes_every_executable_component():
    prereg = json.loads(PREREG.read_text())
    for component in prereg["scripts"].values():
        path = ROOT / component["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == component["sha256"]


def test_preregistration_requires_complete_manual_output_review():
    prereg = json.loads(PREREG.read_text())
    manual = prereg["manual_audit"]
    assert manual["audit_every_selected_vlm_stage_a_output"] is True
    assert manual["audit_every_selected_vlm_stage_b_output"] is True
    assert manual["audit_every_selected_window_blind_first_pass"] is True
    assert manual["audit_every_selected_window_post_reveal"] is True
    assert manual["manual_review_completion_required"] == 1.0


def test_script_supplies_labels_but_never_certifies_pixels():
    prereg = json.loads(PREREG.read_text())
    text = prereg["text_label_contract"]
    assert text["action_label_source"].startswith("normalized_behavior")
    assert text["script_used_for_candidate_anchors_and_label_semantics_not_visual_certification"] is True
    assert prereg["automatic_acceptance"] is False
    assert prereg["corpus_mutation_authorized"] is False


def test_full_source_tiling_and_diverse_routes_are_preregistered():
    prereg = json.loads(PREREG.read_text())
    search = prereg["full_source_search"]
    assert search["complete_tiling_required"] is True
    assert search["window_seconds"] > search["stride_seconds"]
    assert search["maximum_selected_windows_per_source"] == 14
    assert "uniform_temporal_coverage" in search["selection_routes"]
    assert "commentary_statement_anchor" in search["selection_routes"]
