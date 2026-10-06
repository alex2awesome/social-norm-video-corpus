import pytest

from scripts.prepare_instructional_blind_episode_population import align


def test_align_excludes_render_failures_without_mutating_sources() -> None:
    boards = [
        {"item_id": "a", "sheet_path": "a.jpg", "sheet_sha256": "x"},
        {"item_id": "b", "sheet_path": None, "sheet_sha256": None, "error": "missing"},
    ]
    semantics = [{"item_id": "a", "norm": "n1"}, {"item_id": "b", "norm": "n2"}]
    rendered, aligned, summary = align(boards, semantics)
    assert [row["item_id"] for row in rendered] == ["a"]
    assert [row["item_id"] for row in aligned] == ["a"]
    assert summary["render_failures_retained_outside_model_input"] == 1
    assert summary["corpus_mutated"] is False


def test_align_rejects_missing_semantics() -> None:
    with pytest.raises(ValueError, match="lack semantic"):
        align([{"item_id": "a", "sheet_path": "a", "sheet_sha256": "x"}], [])
