from pathlib import Path

import pytest

from scripts.validate_commentary_occurred_event_transfer_selection_v3 import validate_selection


def minimal(index, uid, decision):
    return {
        "transfer_index": index,
        "uid": uid,
        "item_id": f"commentary:{uid}:0",
        "v1_decision": decision,
    }


def test_rejects_overlap_before_dereferencing_lineage_files():
    rows = [minimal(0, "a", "accept"), minimal(1, "b", "reject")]
    with pytest.raises(ValueError, match="overlaps"):
        validate_selection(rows, {"a"}, 1, 1)


def test_rejects_source_duplication_before_dereferencing_lineage_files():
    rows = [minimal(0, "a", "accept"), minimal(1, "a", "reject")]
    with pytest.raises(ValueError, match="source-disjoint"):
        validate_selection(rows, set(), 1, 1)


def test_rejects_wrong_prior_decision_balance_before_lineage_read():
    rows = [minimal(0, "a", "accept"), minimal(1, "b", "accept")]
    with pytest.raises(ValueError, match="balance"):
        validate_selection(rows, set(), 1, 1)
