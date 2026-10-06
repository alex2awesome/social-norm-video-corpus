import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_parallel_managed_audit_jobs import load_jobs, run_jobs


def test_load_jobs_rejects_duplicate_names(tmp_path: Path) -> None:
    path = tmp_path / "jobs.json"
    path.write_text(json.dumps({"jobs": [
        {"name": "same", "argv": ["a"]},
        {"name": "same", "argv": ["b"]},
    ]}))
    with pytest.raises(ValueError, match="unique"):
        load_jobs(path)


def test_run_jobs_returns_sorted_exit_statuses() -> None:
    class Completed:
        def __init__(self, code: int):
            self.returncode = code

    def fake_run(argv: list[str], check: bool) -> Completed:
        assert check is False
        return Completed(0 if argv[0] == "ok" else 2)

    with patch("scripts.run_parallel_managed_audit_jobs.subprocess.run", fake_run):
        results = run_jobs([
            {"name": "z", "argv": ["bad"]},
            {"name": "a", "argv": ["ok"]},
        ])
    assert [(row["name"], row["returncode"]) for row in results] == [
        ("a", 0), ("z", 2)
    ]
