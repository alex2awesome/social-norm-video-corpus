from scripts.evaluate_commentary_title_retrieval import (
    candidate_yes,
    evaluate,
    strict_literal_yes,
)


def result(candidate="yes", action="yes", same_event="yes"):
    return {
        "candidate_event": candidate,
        "title_action_visible": action,
        "actor_action_target_same_event": same_event,
    }


def ledger_row(manual_class, usable, alignment):
    return {
        "manual_class": manual_class,
        "usable_weak_supervision": usable,
        "label_alignment": alignment,
    }


def test_strict_literal_requires_all_three_fields():
    assert candidate_yes(result())
    assert strict_literal_yes(result())
    assert not strict_literal_yes(result(same_event="uncertain"))
    assert not strict_literal_yes(result(candidate="uncertain"))


def test_evaluate_separates_visual_and_speech_targets():
    ledger = {
        1: ledger_row("pass_visual_exact", "yes", "exact"),
        2: ledger_row("pass_situated_speech", "yes", "exact"),
        3: ledger_row("fail_action_occluded", "no", "unknown"),
    }
    report = evaluate(
        ledger,
        {1: result(), 2: result(candidate="no"), 3: result()},
        {1: result(), 2: result(candidate="no"), 3: result(candidate="no")},
        {2},
    )

    dual = report["rules"]["dual_candidate_yes"]
    assert dual["pixel_visible_event"]["precision"] == 1.0
    assert dual["pixel_visible_event"]["recall"] == 1.0
    assert dual["usable_any_tier"]["recall"] == 0.5

    combined = report["rules"]["dual_candidate_or_deixis"]
    assert combined["usable_any_tier"]["recall"] == 1.0
    assert combined["usable_any_tier"]["precision"] == 1.0
