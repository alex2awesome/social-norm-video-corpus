import numpy as np
import pytest

from scripts.materialize_commentary_adaptive_edge_crop import (
    adaptive_vertical_crop,
    crop_filter,
    motion_energy_retained,
)


def test_edge_lines_produce_central_crop() -> None:
    y, height = adaptive_vertical_crop([
        (0.0, 0.02, 1.0, 0.10),
        (0.0, 0.84, 1.0, 0.12),
    ])
    assert y == pytest.approx(0.135)
    assert height == pytest.approx(0.69)
    assert "crop=" in crop_filter(y, height)


def test_central_recurrent_text_abstains() -> None:
    with pytest.raises(ValueError, match="central event region"):
        adaptive_vertical_crop([(0.2, 0.45, 0.5, 0.08)])


def test_too_little_safe_height_abstains() -> None:
    with pytest.raises(ValueError, match="too short"):
        adaptive_vertical_crop([
            (0.0, 0.0, 1.0, 0.30),
            (0.0, 0.68, 1.0, 0.32),
        ])


def test_motion_retention_measures_only_retained_band() -> None:
    first = np.zeros((100, 100, 3), dtype=np.uint8)
    second = first.copy()
    second[40:60, 20:80] = 255
    retained = motion_energy_retained([first, second], 0.30, 0.40)
    assert retained == pytest.approx(1.0)
    missed = motion_energy_retained([first, second], 0.0, 0.20)
    assert missed == pytest.approx(0.0)


def test_invalid_crop_filter_fails_closed() -> None:
    with pytest.raises(ValueError, match="invalid normalized crop"):
        crop_filter(0.8, 0.3)
