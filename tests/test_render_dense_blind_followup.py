import pytest

from scripts.render_dense_blind_followup import select_dense_rows


def blind(item_id, index):
    return {"item_id": item_id, "audit_index": index, "uid": item_id, "pillar": "witnessed"}


def review(item_id, dense, scene="no", behavior="no"):
    return {
        "item_id": item_id,
        "situated_social_scene": scene,
        "concrete_behavior_visible": behavior,
        "affected_person_or_shared_context_visible": "no",
        "presentation_or_context_only": "no",
        "technical_or_formal_only": "no",
        "depiction_type": "uncertain",
        "authenticity": "uncertain",
        "dense_review_required": dense,
        "literal_description": "visible content",
    }


def test_select_dense_rows_preserves_blind_order_and_excludes_non_dense():
    corpus = [
        {"item_id": "a", "source_clip": "a.mp4", "norm": "must not leak"},
        {"item_id": "b", "source_clip": "b.mp4", "title": "must not leak"},
    ]
    selected = select_dense_rows(
        corpus,
        [blind("b", 1), blind("a", 0)],
        [
            review("a", "yes", scene="uncertain"),
            review("b", "no"),
        ],
    )
    assert [blind_row["item_id"] for _, blind_row, _ in selected] == ["a"]


def test_select_dense_rows_requires_exact_review_coverage():
    with pytest.raises(ValueError, match="coverage mismatch"):
        select_dense_rows(
            [{"item_id": "a", "source_clip": "a.mp4"}],
            [blind("a", 0)],
            [],
        )


def test_select_dense_rows_rejects_missing_corpus_item():
    with pytest.raises(ValueError, match="absent from corpus"):
        select_dense_rows(
            [],
            [blind("a", 0)],
            [review("a", "yes", scene="uncertain")],
        )
