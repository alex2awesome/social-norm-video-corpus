import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "report_managed_vllm_usage.py"
)
SPEC = importlib.util.spec_from_file_location(
    "report_managed_vllm_usage",
    SCRIPT,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_summarize_events_counts_gpu_lease_and_skips():
    events = [
        {
            "event": "invocation_started",
            "profile": "qwen",
            "run_id": "a",
            "timestamp_unix": 1,
        },
        {
            "event": "server_started",
            "profile": "qwen",
            "run_id": "a",
            "timestamp_unix": 2,
        },
        {
            "event": "server_ready",
            "profile": "qwen",
            "run_id": "a",
            "timestamp_unix": 3,
            "startup_seconds": 12,
        },
        {
            "event": "client_finished",
            "profile": "qwen",
            "run_id": "a",
            "timestamp_unix": 4,
            "client_seconds": 20,
        },
        {
            "event": "server_unloaded",
            "profile": "qwen",
            "run_id": "a",
            "timestamp_unix": 5,
            "output_records_added": 50,
            "ready_residency_seconds": 21,
            "total_gpu_lease_seconds": 33,
        },
        {
            "event": "invocation_started",
            "profile": "qwen",
            "run_id": "b",
            "timestamp_unix": 6,
        },
        {
            "event": "invocation_skipped",
            "profile": "qwen",
            "run_id": "b",
            "timestamp_unix": 7,
        },
    ]
    row = MODULE.summarize_events(events)["qwen"]
    assert row["invocations"] == 2
    assert row["loads"] == 1
    assert row["completed_clients"] == 1
    assert row["skipped_without_gpu"] == 1
    assert row["records_added"] == 50
    assert row["startup_seconds"] == 12
    assert row["client_seconds"] == 20
    assert row["ready_residency_seconds"] == 21
    assert row["total_gpu_lease_seconds"] == 33


def test_summarize_events_respects_since():
    events = [
        {
            "event": "server_started",
            "profile": "old",
            "run_id": "a",
            "timestamp_unix": 10,
        },
        {
            "event": "server_started",
            "profile": "new",
            "run_id": "b",
            "timestamp_unix": 20,
        },
    ]
    assert set(MODULE.summarize_events(events, since_unix=15)) == {"new"}
