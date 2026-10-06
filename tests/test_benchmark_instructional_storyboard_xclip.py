import numpy as np
from PIL import Image

from scripts.benchmark_instructional_storyboard_xclip import frame_sets


def test_frame_sets_cover_full_storyboard(tmp_path) -> None:
    path = tmp_path / "sheet.jpg"
    Image.new("RGB", (400, 900), "white").save(path)
    videos = frame_sets(path)
    assert len(videos) == 5
    assert all(len(video) == 8 for video in videos)
    assert all(frame.shape[-1] == 3 for video in videos for frame in video)
