import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.run_managed_vlm_audit_bundle import (
    load_jobs,
    render_argv,
    run_bundle,
)


def test_load_jobs_requires_unique_explicit_argv(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "jobs": [
                    {"name": "one", "argv": ["python", "one.py"]},
                    {"name": "two", "argv": ["python", "two.py"]},
                ]
            }
        )
    )
    assert [row["name"] for row in load_jobs(bundle)] == ["one", "two"]


def test_load_jobs_rejects_duplicate_names(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "jobs": [
                    {"name": "same", "argv": ["a"]},
                    {"name": "same", "argv": ["b"]},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="unique"):
        load_jobs(bundle)


def test_render_argv_replaces_only_declared_values() -> None:
    assert render_argv(
        ["client", "--endpoint", "{endpoint}", "--model={model}"],
        "http://localhost/v1",
        "qwen",
    ) == ["client", "--endpoint", "http://localhost/v1", "--model=qwen"]
    with pytest.raises(ValueError, match="unsupported placeholder"):
        render_argv(["--bad={unknown}"], "endpoint", "model")


def test_run_bundle_does_not_use_a_shell_and_continues_after_failure() -> None:
    jobs = [
        {"name": "one", "argv": ["one", "{model}"]},
        {"name": "two", "argv": ["two", "{endpoint}"]},
    ]
    with patch("scripts.run_managed_vlm_audit_bundle.subprocess.run") as run:
        run.side_effect = [
            type("Completed", (), {"returncode": 1})(),
            type("Completed", (), {"returncode": 0})(),
        ]
        results = run_bundle(jobs, "endpoint", "model")
    assert [row["returncode"] for row in results] == [1, 0]
    assert run.call_count == 2
    assert all(call.kwargs == {"check": False} for call in run.call_args_list)
