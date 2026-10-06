import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "score_instructional_v10_consensus.py"
SPEC = importlib.util.spec_from_file_location("v10_consensus", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def qwen(event="yes"):
    return {
        "observable_event": event,
        "same_event_actor_action_target": "yes",
        "event_temporally_localized": "yes",
        "scene_role": "demonstrated_event",
    }


def glm(demo="yes"):
    return {"demo_usable": demo}


def valid():
    return {
        "visual_records_agree_on_literal_event": "yes",
        "performed_demo_supported_by_both": "yes",
        "hard_exclusion": "none",
        "norm_granularity": "adequate_conventional_class",
        "polarity_attaches_to_labeled_actor": "yes",
        "strict_exact_candidate": "yes",
        "usable_after_relabel": "yes",
        "safe_relabel": "avoid insulting coworkers",
        "normalized_performed_behavior": "one coworker insults another",
        "failure_mechanism": "none",
        "evidence": "Both records show one coworker insulting another.",
    }


def test_valid_consensus_positive_parses():
    parsed = MODULE.parse_v10c(json.dumps(valid()), qwen(), glm())
    assert parsed["strict_exact_candidate"] == "yes"


def test_visual_disagreement_fails_positive_closed():
    row = valid()
    row["visual_records_agree_on_literal_event"] = "no"
    parsed = MODULE.parse_v10c(json.dumps(row), qwen(), glm())
    assert parsed["strict_exact_candidate"] == "uncertain"
    assert "inconsistent_strict_positive" in parsed["consistency_repairs"]


def test_upstream_visual_failure_cannot_be_rescued():
    parsed = MODULE.parse_v10c(json.dumps(valid()), qwen("no"), glm())
    assert parsed["strict_exact_candidate"] == "uncertain"
    assert parsed["usable_after_relabel"] == "uncertain"


def test_null_safe_relabel_is_recorded_and_failed_closed():
    row = valid()
    row["safe_relabel"] = None
    parsed = MODULE.parse_v10c(json.dumps(row), qwen(), glm())
    assert parsed["safe_relabel"] == "none"
    assert parsed["safe_relabel_raw"] is None
    assert parsed["strict_exact_candidate"] == "uncertain"
    assert "invalid_string_field" in parsed["consistency_repairs"]


def test_latest_successful_supports_separate_qwen_ledger():
    rows = [
        {"item_id": "x", "model": "qwen", "result": {"observable_event": "yes"}, "error": None},
        {"item_id": "x", "model": "qwen", "result": None, "error": "retry"},
    ]
    assert MODULE.latest_successful(rows)["x"]["model"] == "qwen"
