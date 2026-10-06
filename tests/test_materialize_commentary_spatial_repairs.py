import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_commentary_spatial_repairs.py"
SPEC = importlib.util.spec_from_file_location("spatial_materializer", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def valid_row():
    return {
        "audit_index": 1,
        "variant_id": "001_v2",
        "approval_status": "unreviewed_repair_candidate",
        "source_path": __file__,
        "start_sec": 0,
        "end_sec": 2,
        "crop_norm": [0.1, 0.2, 0.9, 0.8],
    }


def test_plan_requires_unreviewed_status():
    row = valid_row()
    row["approval_status"] = "approved"
    with pytest.raises(ValueError, match="pre-approved"):
        MODULE.validate_plan([row])


def test_plan_rejects_invalid_crop():
    row = valid_row()
    row["crop_norm"] = [0.9, 0.2, 0.1, 0.8]
    with pytest.raises(ValueError, match="crop bounds"):
        MODULE.validate_plan([row])


def test_crop_filter_forces_even_geometry():
    result = MODULE.crop_filter([0.1, 0.2, 0.9, 0.8])
    assert result.startswith("crop=")
    assert "trunc(iw*0.80000000/2)*2" in result
    assert "trunc(ih*0.60000000/2)*2" in result


def test_repair_sheet_contains_both_twelve_frame_panels():
    frames = [
        np.full((24, 32, 3), index, dtype=np.uint8)
        for index in range(24)
    ]
    sheet = MODULE.make_repair_sheet(frames, list(range(24)), 1)
    assert sheet.shape == (1224, 1280, 3)
