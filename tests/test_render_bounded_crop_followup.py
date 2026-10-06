from pathlib import Path

import numpy as np
import pytest

from scripts.render_bounded_crop_followup import crop_frames, validate_plan


def test_crop_frames_uses_normalized_coordinates():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    cropped, pixels = crop_frames([frame], [0.25, 0.2, 0.75, 0.8])
    assert pixels == [50, 20, 150, 80]
    assert cropped[0].shape == (60, 100, 3)


def test_validate_plan_rejects_duplicate_variant(tmp_path: Path):
    source = tmp_path / "video.mp4"
    source.touch()
    row = {
        "variant_id": "a",
        "source_path": str(source),
        "start_sec": 0,
        "end_sec": 1,
        "crop_norm": [0, 0, 1, 1],
    }
    with pytest.raises(ValueError, match="duplicate variant_id"):
        validate_plan([row, row])


def test_validate_plan_rejects_bad_crop(tmp_path: Path):
    source = tmp_path / "video.mp4"
    source.touch()
    row = {
        "variant_id": "a",
        "source_path": str(source),
        "start_sec": 0,
        "end_sec": 1,
        "crop_norm": [0.8, 0, 0.2, 1],
    }
    with pytest.raises(ValueError, match="invalid crop bounds"):
        validate_plan([row])
