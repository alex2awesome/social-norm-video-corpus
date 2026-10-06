import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "score_instructional_v10_alignment.py"
SPEC = importlib.util.spec_from_file_location("v10_alignment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def visual(demo="yes"):
    return {"demo_usable": demo}


def valid():
    return {
        "visual_record_supports_demo": "yes",
        "qualifying_social_scope": "yes",
        "target_behavior_performed_not_merely_described": "yes",
        "actor_action_affected_party_alignment": "yes",
        "proposed_norm_specificity": "exact_concrete",
        "proposed_polarity_relation": "matches_event",
        "visible_event_polarity": "violation",
        "exact_original_usable": "yes",
        "usable_after_relabel": "yes",
        "safe_relabel": "avoid insulting coworkers",
        "failure_mechanism": "none",
        "alignment_evidence": "The coworker performs the labeled insult.",
    }


def test_exact_positive_parses():
    parsed = MODULE.parse_v10b(json.dumps(valid()), visual())
    assert parsed["exact_original_usable"] == "yes"
    assert parsed["usable_after_relabel"] == "yes"


def test_generic_label_cannot_be_exact():
    row = valid()
    row["proposed_norm_specificity"] = "generic_vacuous"
    parsed = MODULE.parse_v10b(json.dumps(row), visual())
    assert parsed["exact_original_usable"] == "uncertain"
    assert "inconsistent_exact_positive" in parsed["consistency_repairs"]


def test_visual_failure_cannot_be_rescued():
    parsed = MODULE.parse_v10b(json.dumps(valid()), visual("no"))
    assert parsed["exact_original_usable"] == "uncertain"
    assert parsed["usable_after_relabel"] == "uncertain"
    assert "visual_gate_not_positive" in parsed["consistency_repairs"]


def test_latest_successful_keeps_last_success_and_ignores_error():
    rows = [
        {"item_id": "a", "result": {"x": 1}, "error": None},
        {"item_id": "a", "result": None, "error": "bad"},
        {"item_id": "a", "result": {"x": 2}, "error": None},
    ]
    assert MODULE.latest_successful(rows)["a"]["result"] == {"x": 2}
