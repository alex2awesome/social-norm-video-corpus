import pytest

from scripts.build_commentary_ocr_line_mask_plan import (
    compatible_line,
    recurrent_line_boxes,
)


def test_compatible_line_matches_same_banner_location():
    assert compatible_line((10, 80, 100, 10), (12, 79, 95, 11), 200, 100)
    assert not compatible_line((10, 80, 100, 10), (10, 20, 100, 10), 200, 100)


def test_recurrent_lines_keep_banner_and_drop_one_frame_false_positive():
    frames = []
    for index in range(8):
        lines = [(10 + index % 2, 80, 110, 10)]
        if index == 0:
            lines.append((130, 10, 40, 20))
        frames.append(lines)
    result = recurrent_line_boxes(frames, 200, 100)
    assert len(result) == 1
    x, y, width, height = result[0]
    assert x < 0.06 and y < 0.80
    assert x + width > 0.59 and y + height > 0.90


def test_recurrent_lines_rejects_oversized_single_component():
    frames = [[(0, 0, 200, 90)] for _ in range(8)]
    with pytest.raises(ValueError, match="no recurrent safe"):
        recurrent_line_boxes(frames, 200, 100)


def test_recurrent_lines_rejects_unsafe_total_union():
    frames = [
        [(0, 0, 200, 10), (0, 20, 200, 10), (0, 40, 200, 10), (0, 60, 200, 10)]
        for _ in range(8)
    ]
    with pytest.raises(ValueError, match="unsafe total OCR mask area"):
        recurrent_line_boxes(
            frames, 200, 100, maximum_box_area=0.20, maximum_total_area=0.20
        )
