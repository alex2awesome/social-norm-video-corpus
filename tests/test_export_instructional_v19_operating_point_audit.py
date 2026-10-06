from scripts.export_instructional_v19_operating_point_audit import export_rows


def test_export_uses_train_threshold_and_joins_manual_outputs() -> None:
    report = {
        "model_results": {
            "low_level": {
                "visual": {
                    "v18_scores": [0.8, 0.2],
                    "best_precision_operating_points": {
                        "3": {
                            "threshold_fit_train_oof": 0.5,
                            "train_oof_metrics": {"precision": 0.7},
                            "v18_transfer_metrics": {"precision": 1.0},
                        }
                    },
                }
            }
        }
    }
    manifest = [
        {"item_id": "a", "audit_cohort": "v18"},
        {"item_id": "b", "audit_cohort": "v18"},
    ]
    manual = [
        {
            "item_id": "a",
            "manual_visual_demo": True,
            "manual_failure_mode": "none",
        },
        {
            "item_id": "b",
            "manual_visual_demo": False,
            "manual_failure_mode": "static",
        },
    ]
    rows, summary = export_rows(
        report, manifest, manual, "low_level", "visual", 3
    )
    assert [row["item_id"] for row in rows] == ["a"]
    assert summary["selected_v18"] == 1
    assert summary["manual_precision"] == 1.0
