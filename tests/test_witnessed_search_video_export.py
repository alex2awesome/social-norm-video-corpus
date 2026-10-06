import importlib.util
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "export_witnessed_search_video_audit",
    ROOT / "scripts/export_witnessed_search_video_audit.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_make_sheet_preserves_all_cells(tmp_path):
    frames = tmp_path / "frames"
    frames.mkdir()
    records = []
    for index in range(5):
        path = frames / f"f{index}.jpg"
        Image.new("RGB", (100, 50), (index * 20, 0, 0)).save(path)
        records.append({"frame_index": index, "timestamp": index + 0.5, "path": path.name})
    out = tmp_path / "sheet.jpg"
    MODULE.make_sheet(records, frames, out)
    with Image.open(out) as image:
        assert image.size == (1600, 540)
