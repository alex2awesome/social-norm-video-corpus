import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_commentary_candidate_clips.py"
SPEC = importlib.util.spec_from_file_location("commentary_candidate_clips", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_plan_must_be_explicitly_unreviewed():
    row = {
        "audit_index": 1,
        "approval_status": "accepted",
        "proposed_start_sec": 1,
        "proposed_end_sec": 2,
        "source_duration_sec": 3,
    }
    with pytest.raises(ValueError, match="pre-approved"):
        MODULE.validate_plan([row])


def test_plan_rejects_duplicate_indices():
    row = {
        "audit_index": 1,
        "approval_status": "unreviewed_candidate",
        "proposed_start_sec": 1,
        "proposed_end_sec": 2,
        "source_duration_sec": 3,
    }
    with pytest.raises(ValueError, match="duplicate"):
        MODULE.validate_plan([row, row])
