import json

from PIL import Image

from scripts.render_hourly_visual_audit_sheets import render_page, shorten


def test_shorten_normalizes_and_truncates():
    assert shorten("  a   b ", 10) == "a b"
    assert shorten("abcdefghij", 5) == "abcd…"


def test_render_page_uses_three_temporal_frames(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    names = []
    for index, color in enumerate(("red", "green", "blue")):
        name = f"f{index}.jpg"
        Image.new("RGB", (64, 64), color).save(frames / name)
        names.append(name)
    target = tmp_path / "sheet.jpg"
    render_page(
        tmp_path,
        [
            {
                "audit_index": 0,
                "pillar": "commentary",
                "title": "title",
                "norm": "norm",
                "statement": "statement",
                "query": "query",
                "frames": names,
            }
        ],
        target,
    )

    assert target.is_file()
    assert Image.open(target).size == (960, 256)
