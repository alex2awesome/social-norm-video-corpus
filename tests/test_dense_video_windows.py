from scripts.export_dense_video_windows import dense_window_key


def test_dense_window_key_is_unique_for_multiple_windows_from_same_source():
    row = {"ordinal": 10, "uid": "dailymotion__example"}

    first = dense_window_key(row, 130.0, 220.0, 0)
    second = dense_window_key(row, 1735.0, 1840.0, 1)

    assert first == "10_dailymotion__example_130_220_00"
    assert second == "10_dailymotion__example_1735_1840_01"
    assert first != second
