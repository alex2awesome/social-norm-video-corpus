import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "evaluate_instructional_v5_corpus_validation.py"
)
SPEC = importlib.util.spec_from_file_location("evaluate_v5", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_wilson_is_bounded_and_contains_rate():
    low, high = MODULE.wilson_95(39, 60)
    assert 0 <= low < 39 / 60 < high <= 1


def test_metric_counts_boolean_judgments():
    result = MODULE.metric(
        [{"manual_positive": True}, {"manual_positive": False}]
    )
    assert result["n"] == 2
    assert result["manual_positive"] == 1
    assert result["rate"] == 0.5


def test_mechanism_family_groups_surface_forms():
    assert (
        MODULE.mechanism_family("talking_heads")
        == "context_only_broll_montage_or_presentation"
    )
    assert (
        MODULE.mechanism_family("formal_safety_rule")
        == "formal_or_procedural_not_informal_social_conduct"
    )
