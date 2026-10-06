from scripts.analyze_instructional_low_level_audit import FEATURES, analyze


def test_analyze_low_level_features_joins_frozen_rows() -> None:
    manifest = [
        {"audit_index": index, "candidate_id": f"c{index}", "item_id": f"i{index}"}
        for index in range(4)
    ]
    ledger = [
        {
            "audit_index": str(index),
            "candidate_id": f"c{index}",
            "visual_demo": "yes" if index < 2 else "no",
        }
        for index in range(4)
    ]
    features = [
        {
            "item_id": f"i{index}",
            "error": None,
            "low_level": {name: float(4 - index) for name in FEATURES},
        }
        for index in range(4)
    ]
    report = analyze(manifest, ledger, features)
    assert report["visual_demos"] == 2
    assert report["features"]["motion_mean"]["raw_average_precision"] == 1.0
    assert report["features"]["motion_mean"]["best_direction"] == "higher"
