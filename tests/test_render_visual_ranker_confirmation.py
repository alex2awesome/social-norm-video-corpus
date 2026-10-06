import numpy as np

from scripts.render_visual_ranker_confirmation import make_dense_sheet


def test_make_dense_sheet_stacks_twelve_frame_panels():
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(36)]
    sheet = make_dense_sheet(frames, [float(i) for i in range(36)], 7)
    assert sheet.shape == (3 * 612, 1280, 3)
