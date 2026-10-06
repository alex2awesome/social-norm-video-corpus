import json
from pathlib import Path

import pytest

from scripts.merge_blind_vlm_manifests import merge


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_merge_requires_exact_disjoint_coverage(tmp_path: Path) -> None:
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    _write(a, [{"item_id": "opaque-1", "audit_index": 1, "error": None}])
    _write(b, [{"item_id": "opaque-0", "audit_index": 0, "error": None}])
    assert [row["audit_index"] for row in merge([a, b], 2)] == [0, 1]

    _write(b, [{"item_id": "opaque-1", "audit_index": 0, "error": None}])
    with pytest.raises(ValueError, match="duplicate item_id"):
        merge([a, b], 2)
