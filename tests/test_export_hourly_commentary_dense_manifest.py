import pytest

from scripts.export_hourly_commentary_dense_manifest import build_items


def test_dense_export_requires_complete_review_and_preserves_source_index():
    manifest = {
        "visual_samples": [
            {
                "audit_index": 2,
                "uid": "x",
                "title": "title",
                "statement": "visible quote",
                "query": "queue cutting",
                "norm": "fairness",
            }
        ]
    }
    reviews = [
        {
            "audit_index": 2,
            "uid": "x",
            "dense_followup": True,
            "source_screen": "potential_event",
            "evidence": "manual",
        }
    ]

    assert build_items(manifest, reviews) == [
        {
            "source_audit_index": 2,
            "uid": "x",
            "title": "title",
            "normalized_behavior": "queue cutting",
            "normalized_norm": "fairness",
            "behavior_evidence_quote": "visible quote",
            "stance_evidence_quote": "visible quote",
            "source_screen": "potential_event",
            "source_evidence": "manual",
        }
    ]


def test_dense_export_refuses_partial_manual_review():
    with pytest.raises(ValueError, match="complete source manifest"):
        build_items(
            {"visual_samples": [{"audit_index": 0, "uid": "x"}]},
            [],
        )
