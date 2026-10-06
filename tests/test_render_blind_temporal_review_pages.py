from PIL import Image

from scripts.render_blind_temporal_review_pages import render_item, render_page


def test_renders_blind_item_and_combined_page(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    names = []
    for index in range(5):
        name = f"{index}.jpg"
        Image.new("RGB", (32, 18), (index * 20, 0, 0)).save(frames / name)
        names.append(name)
    row = {"audit_index": 7, "uid": "source-x", "frames": names}
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    render_item(tmp_path, row, first)
    render_item(tmp_path, row, second)
    page = tmp_path / "page.jpg"
    render_page([first, second], page)

    assert first.is_file()
    assert Image.open(first).size == (1600, 534)
    assert Image.open(page).size == (1600, 1068)
