from scripts.evaluate_commentary_vlm_conjunction import evaluate, predict


def result(depiction="organic_scene"):
    return {
        "behavior_occurs_in_scene": "yes",
        "social_norm_domain": "yes",
        "proposed_norm_supported": "yes",
        "depiction_type": depiction,
    }


def test_predict_requires_organic_depiction():
    assert predict(result())
    assert not predict(result("enacted_scene"))


def test_evaluate_reports_false_positive():
    gold = [{"item_id": "a", "strict_commentary_visual_pass": "no"}]
    vlm = [{"item_id": "a", "error": None, "result": result()}]
    report = evaluate(gold, vlm)
    assert report["fp"] == 1
    assert report["precision"] == 0.0
