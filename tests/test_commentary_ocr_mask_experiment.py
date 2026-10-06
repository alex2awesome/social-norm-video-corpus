import numpy as np
import pytest

from scripts.build_commentary_ocr_mask_plan import persistent_text_boxes
from scripts.materialize_commentary_ocr_mask_experiment import mask_filter, validate_plan


def test_persistent_text_boxes_keeps_recurrent_region_not_one_frame_noise():
    recurrent = (10, 80, 100, 10)
    boxes = [
        [recurrent, (140, 10, 10, 10)] if index == 0 else [recurrent]
        for index in range(8)
    ]
    result = persistent_text_boxes(boxes, 200, 100)
    assert result
    assert any(y < 0.82 and y + height > 0.90 for _, y, _, height in result)
    assert not any(x > 0.65 and y < 0.25 for x, y, _, _ in result)


def test_persistent_text_boxes_requires_multiple_frames():
    assert persistent_text_boxes([[(5, 5, 20, 10)]] + [[] for _ in range(7)], 100, 100) == []


def plan():
    return [{
        "audit_index": 0,
        "candidate_id": "parent--ocr_mask_v1",
        "parent_candidate_id": "parent",
        "variant": "ocr_mask_v1",
        "mask_boxes_normalized": [[0.1, 0.8, 0.7, 0.1]],
        "source_path": "/data/a.mp4",
        "source_start_sec": 1.0,
        "source_end_sec": 5.0,
    }]


def test_validate_plan_accepts_normalized_boxes():
    assert validate_plan(plan())[0]["candidate_id"] == "parent--ocr_mask_v1"


def test_validate_plan_accepts_guarded_v2_variant():
    rows = plan()
    rows[0]["variant"] = "ocr_line_mask_v2"
    assert validate_plan(rows)[0]["variant"] == "ocr_line_mask_v2"


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("variant", "manual", "unexpected variant"),
        ("mask_boxes_normalized", [], "missing mask boxes"),
        ("mask_boxes_normalized", [[0.8, 0, 0.3, 0.2]], "exceeds frame"),
        ("source_end_sec", 0.5, "invalid temporal bounds"),
    ],
)
def test_validate_plan_fails_closed(field, value, match):
    rows = plan()
    rows[0][field] = value
    with pytest.raises(ValueError, match=match):
        validate_plan(rows)


def test_mask_filter_is_deterministic_and_uses_black_fill():
    assert mask_filter([[0.1, 0.8, 0.7, 0.1]]) == (
        "drawbox=x=iw*0.100000:y=ih*0.800000:w=iw*0.700000:"
        "h=ih*0.100000:color=black:t=fill"
    )


def test_mask_filter_rejects_empty_input():
    with pytest.raises(ValueError, match="no OCR boxes"):
        mask_filter([])
