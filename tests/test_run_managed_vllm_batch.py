import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_managed_vllm_batch.py"
SPEC = importlib.util.spec_from_file_location("run_managed_vllm_batch", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_load_profile_accepts_profiles_envelope(tmp_path):
    config = tmp_path / "profiles.json"
    config.write_text(
        json.dumps(
            {
                "profiles": {
                    "tiny": {
                        "endpoint": "http://127.0.0.1:9999/v1",
                        "model": "tiny",
                        "server_argv": ["server"],
                        "lock_file": str(tmp_path / "lock"),
                        "log_file": str(tmp_path / "log"),
                    }
                }
            }
        )
    )
    profile = MODULE.load_profile(config, "tiny")
    assert profile["model"] == "tiny"


def test_endpoint_has_model_checks_exact_id(monkeypatch):
    response = Mock()
    response.__enter__ = Mock(return_value=response)
    response.__exit__ = Mock(return_value=False)
    response.read.return_value = b'{"data":[{"id":"wanted"}]}'
    monkeypatch.setattr(MODULE.urllib.request, "urlopen", Mock(return_value=response))
    assert MODULE.endpoint_has_model("http://127.0.0.1:1/v1", "wanted")
    assert not MODULE.endpoint_has_model("http://127.0.0.1:1/v1", "other")


def test_wait_for_server_fails_if_process_exits():
    process = Mock(spec=subprocess.Popen)
    process.poll.return_value = 7
    try:
        MODULE.wait_for_server(
            "http://127.0.0.1:1/v1",
            "tiny",
            process,
            MODULE.time.monotonic() + 5,
            0.01,
        )
    except RuntimeError as exc:
        assert "code 7" in str(exc)
    else:
        raise AssertionError("expected early server exit to fail")


def test_parse_args_preserves_client_command():
    args = MODULE.parse_args(
        [
            "--profiles",
            "profiles.json",
            "--profile",
            "tiny",
            "--",
            sys.executable,
            "-c",
            "print('ok')",
        ]
    )
    assert args.client_argv[:2] == [sys.executable, "-c"]


def test_pending_batch_items_subtracts_resumable_output(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "output.jsonl"
    manifest.write_text('{"id": 1}\n\n{"id": 2}\n{"id": 3}\n')
    output.write_text('{"id": 1}\n')

    assert MODULE.count_jsonl_records(manifest) == 3
    assert MODULE.pending_batch_items(manifest, output) == 2


def test_pending_batch_items_retries_error_rows_and_accepts_later_success(
    tmp_path,
):
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "output.jsonl"
    manifest.write_text(
        '{"item_id":"a"}\n{"item_id":"b"}\n{"item_id":"c"}\n'
    )
    output.write_text(
        '{"item_id":"a","error":null,"result":{}}\n'
        '{"item_id":"b","error":"invalid schema","result":null}\n'
    )

    assert MODULE.pending_batch_items(manifest, output) == 2

    with output.open("a") as handle:
        handle.write('{"item_id":"b","error":null,"result":{}}\n')
    assert MODULE.pending_batch_items(manifest, output) == 1


def test_pending_batch_items_prefers_candidate_id_over_shared_item_id(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "output.jsonl"
    manifest.write_text(
        '{"candidate_id":"a:0","item_id":"a"}\n'
        '{"candidate_id":"a:1","item_id":"a"}\n'
        '{"candidate_id":"a:2","item_id":"a"}\n'
    )
    output.write_text(
        '{"candidate_id":"a:0","item_id":"a","error":null}\n'
    )

    assert MODULE.pending_batch_items(manifest, output) == 2


def test_pending_batch_items_falls_back_for_unkeyed_legacy_rows(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    output = tmp_path / "output.jsonl"
    manifest.write_text('{"value":1}\n{"value":2}\n')
    output.write_text('{"value":1}\n')

    assert MODULE.pending_batch_items(manifest, output) == 1


def test_wait_for_batch_starts_immediately_at_size_threshold(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id": 1}\n{"id": 2}\n')

    pending, reason = MODULE.wait_for_batch(manifest, None, 2, 3600, 60)

    assert pending == 2
    assert reason == "size_threshold"


def test_wait_for_batch_runs_small_remainder_at_deadline(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{"id": 1}\n')

    pending, reason = MODULE.wait_for_batch(manifest, None, 50, 0, 60)

    assert pending == 1
    assert reason == "max_delay"


def test_wait_for_batch_skips_empty_manifest_at_deadline(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    pending, reason = MODULE.wait_for_batch(manifest, None, 50, 0, 60)

    assert pending == 0
    assert reason == "empty"


def test_checked_in_sk2_profiles_are_complete_and_use_distinct_resources():
    config = SCRIPT.parents[1] / "config" / "sk2_audit_vllm_profiles.json"
    names = ("qwen3_vl_8b", "glm_4_6v_flash", "gemma_3_27b_it")
    profiles = [MODULE.load_profile(config, name) for name in names]

    assert len({profile["endpoint"] for profile in profiles}) == len(profiles)
    assert len({profile["gpu"] for profile in profiles}) == len(profiles)
    assert len({profile["lock_file"] for profile in profiles}) == len(profiles)
    assert all(profile["min_free_mib"] >= 120000 for profile in profiles)
    assert all(
        profile["env"]["FLASHINFER_WORKSPACE_BASE"].startswith(
            "/lfs/skampere2/"
        )
        for profile in profiles
    )
    assert profiles[2]["env"]["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_checked_in_sk3_shared_cache_profile_is_offline_and_dynamic():
    config = SCRIPT.parents[1] / "config" / "sk3_audit_vllm_profiles.json"
    profile = MODULE.load_profile(config, "qwen3_vl_8b_low_impact")

    assert profile["env"]["HF_HOME"] == "/lfs/skampere3/0/shared_hf_cache"
    assert profile["env"]["HF_HUB_OFFLINE"] == "1"
    assert profile["env"]["TRANSFORMERS_OFFLINE"] == "1"
    assert profile["gpu"] == 7
    assert profile["gpu_candidates"] == [0, 5, 6, 7]
    assert profile["min_free_mib"] >= 90000
    assert profile["state_file"].startswith("/lfs/skampere3/")


def test_wait_for_gpu_selects_freest_allowed_candidate(monkeypatch):
    free = {0: 100000, 5: 150000, 6: 120000, 7: 13000}
    monkeypatch.setattr(MODULE, "gpu_free_mib", lambda gpu: free[gpu])

    selected = MODULE.wait_for_gpu(
        {"gpu_candidates": [0, 5, 6, 7], "min_free_mib": 90000},
        MODULE.time.monotonic() + 1,
    )

    assert selected == 5


def test_gpu_candidates_rejects_implicit_or_duplicate_values():
    assert MODULE.configured_gpu_candidates({"gpu": 7}) == [7]
    try:
        MODULE.configured_gpu_candidates({"gpu_candidates": [5, 5]})
    except ValueError as exc:
        assert "unique" in str(exc)
    else:
        raise AssertionError("duplicate GPU allow-list should fail")


def test_run_client_stops_child_when_wait_is_interrupted(monkeypatch):
    process = Mock(spec=subprocess.Popen)
    process.wait.side_effect = MODULE.TerminationRequested(15)
    process.poll.return_value = None
    popen = Mock(return_value=process)
    stop = Mock()
    monkeypatch.setattr(MODULE.subprocess, "Popen", popen)
    monkeypatch.setattr(MODULE, "stop_process_group", stop)

    try:
        MODULE.run_client(["client"], 17)
    except MODULE.TerminationRequested as exc:
        assert exc.signum == 15
    else:
        raise AssertionError("expected termination to propagate")

    popen.assert_called_once_with(
        ["client"],
        start_new_session=True,
        preexec_fn=MODULE.set_parent_death_signal,
    )
    stop.assert_called_once_with(process, 17)


def test_parent_death_signal_is_noop_off_linux(monkeypatch):
    monkeypatch.setattr(MODULE.sys, "platform", "darwin")
    libc = Mock()
    monkeypatch.setattr(MODULE.ctypes, "CDLL", libc)

    MODULE.set_parent_death_signal()

    libc.assert_not_called()


def test_parent_death_signal_requests_sigterm_on_linux(monkeypatch):
    monkeypatch.setattr(MODULE.sys, "platform", "linux")
    monkeypatch.setattr(MODULE.os, "getppid", Mock(return_value=123))
    prctl = Mock(return_value=0)
    library = Mock(prctl=prctl)
    monkeypatch.setattr(MODULE.ctypes, "CDLL", Mock(return_value=library))

    MODULE.set_parent_death_signal()

    prctl.assert_called_once_with(1, MODULE.signal.SIGTERM, 0, 0, 0)


def test_append_event_writes_parseable_jsonl(tmp_path):
    events = tmp_path / "events.jsonl"
    MODULE.append_event(events, "server_ready", "run-1", profile="tiny", gpu=7)

    row = json.loads(events.read_text())
    assert row["event"] == "server_ready"
    assert row["run_id"] == "run-1"
    assert row["profile"] == "tiny"
    assert row["gpu"] == 7
    assert row["timestamp_unix"] > 0
