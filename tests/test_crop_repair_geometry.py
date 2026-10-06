import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_instructional_crop_repair_audit.py"
SPEC = importlib.util.spec_from_file_location("crop_repair", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CropGeometryTest(unittest.TestCase):
    def test_explicit_geometry_is_preserved(self):
        self.assertEqual(
            MODULE.resolve_crop_geometry(640, 360, None, "296:256:332:44"),
            ("296:256:332:44", 296, 256),
        )

    def test_explicit_geometry_must_fit_source(self):
        with self.assertRaisesRegex(ValueError, "exceeds"):
            MODULE.resolve_crop_geometry(640, 360, None, "400:300:300:100")

    def test_bottom_crop_remains_backward_compatible(self):
        self.assertEqual(
            MODULE.resolve_crop_geometry(640, 360, 0.1, None),
            ("640:324:0:0", 640, 324),
        )

    def test_exactly_one_crop_mode_is_required(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MODULE.resolve_crop_geometry(640, 360, None, None)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            MODULE.resolve_crop_geometry(640, 360, 0.1, "296:256:332:44")


if __name__ == "__main__":
    unittest.main()
