import cv2
import numpy as np
import pytest

from scripts.materialize_commentary_motion_crop_experiment import (
    crop_filter,
    estimate_motion_bbox,
)


def moving_scene():
    frames = []
    for index in range(12):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        cv2.rectangle(frame, (40 + index * 3, 30), (65 + index * 3, 75), (255, 255, 255), -1)
        # A static label band must not attract the motion crop.
        cv2.rectangle(frame, (0, 82), (159, 99), (180, 180, 180), -1)
        frames.append(frame)
    return frames


def test_motion_bbox_excludes_static_caption_band_and_contains_actor_path():
    x, y, width, height = estimate_motion_bbox(moving_scene())
    assert x <= 40 / 160
    assert x + width >= 98 / 160
    assert y <= 30 / 100
    assert y + height >= 75 / 100
    assert y + height < 0.95


def test_motion_bbox_requires_enough_frames():
    with pytest.raises(ValueError, match="at least eight"):
        estimate_motion_bbox(moving_scene()[:7])


def test_motion_bbox_rejects_static_video():
    frames = [np.zeros((100, 160, 3), dtype=np.uint8) for _ in range(10)]
    with pytest.raises(ValueError, match="insufficient temporal motion"):
        estimate_motion_bbox(frames)


def test_motion_bbox_rejects_dimension_changes():
    frames = moving_scene()
    frames[-1] = np.zeros((90, 160, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="share dimensions"):
        estimate_motion_bbox(frames)


def test_crop_filter_is_normalized_and_deterministic():
    result = crop_filter((0.1, 0.2, 0.7, 0.6))
    assert result == (
        "crop=trunc(iw*0.700000/2)*2:trunc(ih*0.600000/2)*2:"
        "trunc(iw*0.100000/2)*2:trunc(ih*0.200000/2)*2"
    )


@pytest.mark.parametrize(
    "bbox", [(-0.1, 0, 0.5, 0.5), (0, 0, 0, 0.5), (0.8, 0, 0.3, 0.5)]
)
def test_crop_filter_rejects_invalid_boxes(bbox):
    with pytest.raises(ValueError):
        crop_filter(bbox)
