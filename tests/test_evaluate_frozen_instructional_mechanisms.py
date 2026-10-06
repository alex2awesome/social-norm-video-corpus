import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "evaluate_frozen_instructional_mechanisms.py"
)
SPEC = importlib.util.spec_from_file_location("mechanisms", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_strict_v5_requires_every_visual_requirement():
    result = {
        "social_norm_domain": "yes",
        "behavior_occurs_in_scene": "yes",
        "affected_party_or_shared_setting_visible": "yes",
        "situated_interaction_complete": "yes",
        "usable_demo_after_relabel": "yes",
        "proposed_norm_supported": "yes",
    }
    assert MODULE.strict_v5(result)
    assert MODULE.strict_v5(result, exact=True)
    result["behavior_occurs_in_scene"] = "no"
    assert not MODULE.strict_v5(result)


def test_text_strict_excludes_procedure_and_requires_structure():
    result = {
        "social_norm_candidate": "yes",
        "concrete_behavior_named": "yes",
        "affected_other_or_shared_setting_named": "yes",
        "norm_type": "technical_procedure",
    }
    assert MODULE.text_pass(result)
    assert not MODULE.text_pass(result, strict=True)
    result["norm_type"] = "tacit_interpersonal"
    assert MODULE.text_pass(result, strict=True)


def test_frozen_threshold_directions():
    assert MODULE.apply_frozen_threshold(
        0.6, {"direction": "ge", "threshold": 0.5}
    )
    assert MODULE.apply_frozen_threshold(
        0.4, {"direction": "le", "threshold": 0.5}
    )
