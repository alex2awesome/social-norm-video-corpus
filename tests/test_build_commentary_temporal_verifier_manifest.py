import numpy as np

from scripts.build_commentary_temporal_verifier_manifest import (
    _valid_ocr_word,
    bounded_sampling_fps,
    easyocr_detect_boxes,
    is_overlay_band_box,
    mask_text_boxes,
    merge_overlapping_boxes,
    persistent_boxes,
    resize_bounded,
    resize_width,
    select_easyocr_caption_rows,
    select_overlay_text_boxes,
    tile_frames,
    uses_portrait_all_screen_ocr,
)


def test_ocr_word_filter_keeps_single_digit_but_not_single_letter_noise():
    assert _valid_ocr_word("1,", confidence=90, threshold=50)
    assert not _valid_ocr_word("I", confidence=90, threshold=50)


def test_resize_width_preserves_aspect_ratio():
    frame = np.zeros((40, 80, 3), dtype=np.uint8)
    resized = resize_width(frame, 120)
    assert resized.shape == (60, 120, 3)


def test_resize_bounded_limits_portrait_height_without_distortion():
    frame = np.zeros((160, 90, 3), dtype=np.uint8)
    resized = resize_bounded(frame, maximum_width=120, maximum_height=80)
    assert resized.shape == (80, 45, 3)


def test_resize_bounded_limits_landscape_width_without_distortion():
    frame = np.zeros((90, 160, 3), dtype=np.uint8)
    resized = resize_bounded(frame, maximum_width=80, maximum_height=120)
    assert resized.shape == (45, 80, 3)


def test_bounded_sampling_fps_caps_long_window_but_not_short_window():
    assert bounded_sampling_fps(30, 4, 96) == 3.2
    assert bounded_sampling_fps(10, 4, 96) == 4
    assert bounded_sampling_fps(30, 4, None) == 4


def test_mask_text_boxes_changes_only_selected_rectangle():
    frame = np.full((12, 16, 3), 200, dtype=np.uint8)
    frame[4:6, 6:8] = (10, 20, 30)
    masked = mask_text_boxes(frame, [(4, 3, 6, 4)])
    assert np.array_equal(masked[:3], frame[:3])
    assert np.all(masked[3:7, 4:10] == 200)
    assert np.array_equal(masked[7:], frame[7:])


def test_overlapping_ocr_boxes_are_merged():
    merged = merge_overlapping_boxes([(10, 10, 20, 10), (11, 11, 18, 8)])
    assert merged == [(10, 10, 20, 10)]


def test_overlay_filter_excludes_scene_interior():
    assert is_overlay_band_box((10, 5, 20, 10), frame_height=100)
    assert is_overlay_band_box((10, 80, 20, 10), frame_height=100)
    assert not is_overlay_band_box((10, 40, 20, 10), frame_height=100)


def test_aggressive_all_screen_ocr_is_restricted_to_portrait_frames():
    assert uses_portrait_all_screen_ocr(200, 360, enabled=True)
    assert not uses_portrait_all_screen_ocr(480, 269, enabled=True)
    assert not uses_portrait_all_screen_ocr(200, 360, enabled=False)


def test_persistent_boxes_reject_one_frame_false_positive():
    boxes = persistent_boxes(
        [
            [(10, 10, 20, 8)],
            [],
            [(100, 60, 15, 7)],
        ],
        minimum_frames=2,
    )
    assert boxes == []


def test_persistent_boxes_keep_overlapping_region_across_frames():
    boxes = persistent_boxes(
        [
            [(10, 70, 40, 8)],
            [(12, 69, 39, 9)],
            [],
        ],
        minimum_frames=2,
    )
    assert boxes == [(10, 69, 41, 9)]


def test_overlay_selector_keeps_dense_caption_run():
    boxes = [
        (20, 80, 30, 10),
        (54, 81, 35, 9),
        (94, 80, 40, 10),
    ]
    assert select_overlay_text_boxes(boxes, 200, 100) == [(0, 76, 200, 18)]


def test_overlay_selector_rejects_isolated_scene_labels():
    boxes = [
        (10, 80, 20, 10),
        (100, 80, 20, 10),
        (150, 20, 35, 30),
    ]
    assert select_overlay_text_boxes(boxes, 200, 100) == []


def test_overlay_selector_keeps_one_wide_detected_caption():
    assert select_overlay_text_boxes([(20, 40, 110, 12)], 200, 100) == [
        (0, 36, 200, 20)
    ]


def test_easyocr_detector_converts_horizontal_box_to_padded_rectangle():
    class Reader:
        def detect(self, _frame, **_kwargs):
            return ([[[20, 120, 40, 60]]], [[]])

    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    assert easyocr_detect_boxes(frame, Reader()) == [(16, 36, 108, 28)]


def test_easyocr_caption_rows_cover_multiline_persistent_title():
    rows = select_easyocr_caption_rows(
        [(5, 201, 190, 87), (11, 241, 178, 38)],
        frame_width=202,
        frame_height=360,
    )
    assert rows == [(0, 184, 202, 121)]


def test_tile_frames_preserves_chronological_cell_order():
    frames = [
        np.full((2, 3, 3), value, dtype=np.uint8)
        for value in (10, 20, 30, 40, 50)
    ]
    pages = tile_frames(frames, columns=2, rows=2, padding=1, margin=1)
    assert len(pages) == 2
    first = pages[0]
    assert first[1, 1, 0] == 10
    assert first[1, 5, 0] == 20
    assert first[4, 1, 0] == 30
    assert first[4, 5, 0] == 40
    assert pages[1][1, 1, 0] == 50
