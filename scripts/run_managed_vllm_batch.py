#!/usr/bin/env python3
"""Run one batch while a vLLM server is leased, then release its GPU.

The wrapper deliberately manages *batches*, not individual HTTP requests.  A
batch waits on a profile-specific file lock, waits for the configured GPU to
have enough free memory, starts vLLM, verifies that the expected model is
served, runs the client command, and shuts vLLM down even when the client
fails.  This keeps occasional audit models out of GPU memory between batches
without paying model startup cost for every item.
"""

from __future__ import annotations

import argparse
import ctypes
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any
import urllib.error
import urllib.request


class TerminationRequested(Exception):
    """Raised by a signal handler so normal cleanup can release the GPU."""

    def __init__(self, signum: int):
        super().__init__(f"received signal {signum}")
        self.signum = signum


def raise_on_termination(signum: int, _frame: object) -> None:
    raise TerminationRequested(signum)


def load_profile(path: Path, name: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    profiles = payload.get("profiles", payload)
    if name not in profiles:
        raise ValueError(f"unknown profile {name!r}; choices: {sorted(profiles)}")
    profile = profiles[name]
    required = {"endpoint", "model", "server_argv", "lock_file", "log_file"}
    missing = sorted(required - profile.keys())
    if missing:
        raise ValueError(f"profile {name!r} is missing: {', '.join(missing)}")
    if not isinstance(profile["server_argv"], list) or not profile["server_argv"]:
        raise ValueError(f"profile {name!r} server_argv must be a non-empty list")
    return profile


def models_url(endpoint: str) -> str:
    return endpoint.rstrip("/") + "/models"


def endpoint_has_model(endpoint: str, model: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(models_url(endpoint), timeout=timeout) as response:
            payload = json.load(response)
    except (OSError, ValueError, urllib.error.URLError):
        return False
    return any(row.get("id") == model for row in payload.get("data", []))


def gpu_free_mib(index: int) -> int:
    command = [
        "nvidia-smi",
        f"--id={index}",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
    ]
    value = subprocess.check_output(command, text=True).strip().splitlines()[0]
    return int(value)


def count_jsonl_records(path: Path | None) -> int:
    """Count non-empty JSONL records without loading a large manifest."""
    if path is None or not path.exists():
        return 0
    with path.open("rb") as handle:
        return sum(bool(line.strip()) for line in handle)


def jsonl_keyed_rows(
    path: Path | None,
    *,
    successful_only: bool,
) -> set[tuple[str, str]] | None:
    """Read stable record keys, or return None when a file is not keyable."""
    if path is None or not path.exists():
        return set()
    keys: set[tuple[str, str]] = set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSONL"
                ) from exc
            if not isinstance(row, dict):
                return None
            key = next(
                (
                    (field, str(row[field]))
                    for field in ("candidate_id", "item_id", "id", "uid")
                    if row.get(field) not in (None, "")
                ),
                None,
            )
            if key is None:
                return None
            if successful_only and row.get("error") not in (None, ""):
                continue
            keys.add(key)
    return keys


def pending_batch_items(manifest: Path, completed_output: Path | None) -> int:
    """Return the resumable batch size.

    Prefer stable JSONL keys and count only successful output rows.  This
    preserves append-only error attempts while still scheduling their retry.
    Fall back to row counts for legacy manifests that have no common key.
    """
    manifest_keys = jsonl_keyed_rows(manifest, successful_only=False)
    output_keys = jsonl_keyed_rows(completed_output, successful_only=True)
    if manifest_keys is not None and output_keys is not None:
        manifest_fields = {field for field, _value in manifest_keys}
        output_fields = {field for field, _value in output_keys}
        if not manifest_keys:
            return 0
        if not output_keys or manifest_fields == output_fields:
            return len(manifest_keys - output_keys)
    return max(
        0,
        count_jsonl_records(manifest) - count_jsonl_records(completed_output),
    )


def wait_for_batch(
    manifest: Path,
    completed_output: Path | None,
    min_items: int,
    max_delay_seconds: float,
    poll_seconds: float,
) -> tuple[int, str]:
    """Wait CPU-only until a batch is large enough or its latency budget expires."""
    started = time.monotonic()
    while True:
        pending = pending_batch_items(manifest, completed_output)
        elapsed = time.monotonic() - started
        if pending >= min_items:
            return pending, "size_threshold"
        if elapsed >= max_delay_seconds:
            if pending:
                return pending, "max_delay"
            return 0, "empty"

        remaining = max_delay_seconds - elapsed
        wait = min(poll_seconds, remaining)
        print(
            f"managed-vllm: delaying GPU load: {pending}/{min_items} pending "
            f"items; {remaining:.0f}s until latency deadline",
            flush=True,
        )
        time.sleep(wait)


def configured_gpu_candidates(profile: dict[str, Any]) -> list[int]:
    """Return an ordered, explicit GPU allow-list for this profile."""
    values = profile.get("gpu_candidates")
    if values is None:
        return [] if "gpu" not in profile else [int(profile["gpu"])]
    if (
        not isinstance(values, list)
        or not values
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("gpu_candidates must be a nonempty list of unique nonnegative integers")
    return list(values)


def wait_for_gpu(profile: dict[str, Any], deadline: float) -> int | None:
    candidates = configured_gpu_candidates(profile)
    if not candidates:
        return None
    required = int(profile.get("min_free_mib", 0))
    poll = float(profile.get("gpu_poll_seconds", 30))
    while True:
        free_by_gpu = {gpu: gpu_free_mib(gpu) for gpu in candidates}
        gpu = max(candidates, key=lambda value: (free_by_gpu[value], -candidates.index(value)))
        free = free_by_gpu[gpu]
        if free >= required:
            print(
                f"managed-vllm: GPU {gpu} ready ({free} MiB free; "
                f"need {required} MiB; candidates={free_by_gpu})",
                flush=True,
            )
            return gpu
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"no allowed GPU has {required} MiB free; candidates={free_by_gpu}"
            )
        print(
            f"managed-vllm: waiting for allowed GPU: best={gpu} with {free} MiB; "
            f"need {required} MiB; candidates={free_by_gpu}",
            flush=True,
        )
        time.sleep(min(poll, max(0.0, deadline - time.monotonic())))


def wait_for_server(
    endpoint: str,
    model: str,
    process: subprocess.Popen[bytes],
    deadline: float,
    poll_seconds: float,
) -> None:
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(f"vLLM exited during startup with code {return_code}")
        if endpoint_has_model(endpoint, model):
            print(f"managed-vllm: {model} is healthy at {endpoint}", flush=True)
            return
        time.sleep(poll_seconds)
    raise TimeoutError(f"{model} did not become healthy at {endpoint}")


def stop_process_group(process: subprocess.Popen[bytes], timeout: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=timeout)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def set_parent_death_signal() -> None:
    """Ask Linux to terminate a child if this wrapper disappears abruptly.

    This runs in the forked child immediately before exec.  Process groups
    remain the normal graceful-cleanup mechanism; PR_SET_PDEATHSIG is a
    last-resort guard against an orphaned resident model after wrapper death.
    """
    if sys.platform != "linux":
        return
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    # Linux prctl(PR_SET_PDEATHSIG, SIGTERM).
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Close the fork/prctl race: the parent may have died before prctl ran.
    if os.getppid() != parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def append_event(path: Path, event: str, run_id: str, **fields: Any) -> None:
    """Append one process-safe lifecycle record for utilization accounting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "event": event,
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timestamp_unix": time.time(),
        **fields,
    }
    encoded = json.dumps(payload, sort_keys=True) + "\n"
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(encoded)
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


def write_state(
    path: Path,
    profile_name: str,
    process: subprocess.Popen[bytes],
    run_id: str,
    gpu: int | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pid": process.pid,
        "profile": profile_name,
        "run_id": run_id,
        "started_at_unix": time.time(),
        "gpu": gpu,
    }
    path.write_text(json.dumps(payload, sort_keys=True) + "\n")


def run_client(client_argv: list[str], shutdown_timeout: float) -> int:
    """Run the batch client and prevent it from surviving wrapper cancellation."""
    process = subprocess.Popen(
        client_argv,
        start_new_session=True,
        preexec_fn=set_parent_death_signal,
    )
    try:
        return process.wait()
    finally:
        stop_process_group(process, shutdown_timeout)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lease a vLLM server for one queued batch"
    )
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--gpu-wait-timeout", type=float, default=86400)
    parser.add_argument("--shutdown-timeout", type=float, default=60)
    parser.add_argument("--startup-poll-seconds", type=float, default=5)
    parser.add_argument(
        "--batch-manifest",
        type=Path,
        help=(
            "optional accumulating JSONL manifest; no GPU is touched until it "
            "has enough pending records or the delay budget expires"
        ),
    )
    parser.add_argument(
        "--batch-output",
        type=Path,
        help=(
            "optional append-only JSONL output used to subtract already "
            "attempted records from --batch-manifest"
        ),
    )
    parser.add_argument("--min-batch-items", type=int, default=50)
    parser.add_argument("--max-batch-delay-seconds", type=float, default=3600)
    parser.add_argument("--batch-poll-seconds", type=float, default=60)
    parser.add_argument(
        "client_argv",
        nargs=argparse.REMAINDER,
        help="client command after --",
    )
    args = parser.parse_args(argv)
    if args.client_argv and args.client_argv[0] == "--":
        args.client_argv = args.client_argv[1:]
    if not args.client_argv:
        parser.error("a client command is required after --")
    if args.batch_output is not None and args.batch_manifest is None:
        parser.error("--batch-output requires --batch-manifest")
    if args.min_batch_items < 1:
        parser.error("--min-batch-items must be at least 1")
    if args.max_batch_delay_seconds < 0:
        parser.error("--max-batch-delay-seconds cannot be negative")
    if args.batch_poll_seconds <= 0:
        parser.error("--batch-poll-seconds must be positive")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    profile = load_profile(args.profiles, args.profile)
    run_id = f"{os.getpid()}-{time.time_ns()}"
    lock_path = Path(profile["lock_file"])
    state_path = Path(
        profile.get("state_file", f"{profile['lock_file']}.state.json")
    )
    log_path = Path(profile["log_file"])
    events_path = Path(
        profile.get(
            "events_file",
            str(lock_path.parent / "managed_vllm_events.jsonl"),
        )
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    invocation_started = time.monotonic()
    manifest_before = (
        count_jsonl_records(args.batch_manifest)
        if args.batch_manifest is not None
        else None
    )
    output_before = (
        count_jsonl_records(args.batch_output)
        if args.batch_output is not None
        else None
    )
    append_event(
        events_path,
        "invocation_started",
        run_id,
        profile=args.profile,
        manifest=str(args.batch_manifest) if args.batch_manifest else None,
        output=str(args.batch_output) if args.batch_output else None,
        manifest_records=manifest_before,
        output_records=output_before,
        min_batch_items=args.min_batch_items,
        max_batch_delay_seconds=args.max_batch_delay_seconds,
    )

    old_handlers = {
        signum: signal.signal(signum, raise_on_termination)
        for signum in (signal.SIGTERM, signal.SIGHUP)
    }
    try:
        batch_reason: str | None = None
        if args.batch_manifest is not None:
            pending, batch_reason = wait_for_batch(
                args.batch_manifest,
                args.batch_output,
                args.min_batch_items,
                args.max_batch_delay_seconds,
                args.batch_poll_seconds,
            )
            if not pending:
                print(
                    "managed-vllm: batch deadline reached with no pending "
                    "records; exiting without touching a GPU",
                    flush=True,
                )
                append_event(
                    events_path,
                    "invocation_skipped",
                    run_id,
                    profile=args.profile,
                    reason="empty",
                    queue_wait_seconds=time.monotonic() - invocation_started,
                )
                return 0
            print(
                f"managed-vllm: batch ready ({pending} pending; "
                f"reason={batch_reason})",
                flush=True,
            )
            append_event(
                events_path,
                "batch_ready",
                run_id,
                profile=args.profile,
                pending_items=pending,
                reason=batch_reason,
                queue_wait_seconds=time.monotonic() - invocation_started,
            )

        with lock_path.open("a+") as lock_handle:
            print(f"managed-vllm: waiting for lease {args.profile}", flush=True)
            fcntl.flock(lock_handle, fcntl.LOCK_EX)
            print(f"managed-vllm: acquired lease {args.profile}", flush=True)
            lease_acquired = time.monotonic()
            append_event(
                events_path,
                "lease_acquired",
                run_id,
                profile=args.profile,
                invocation_wait_seconds=lease_acquired - invocation_started,
            )

            # A previous lease may have completed this resumable output while
            # this invocation waited.  Recheck before querying GPU state.
            if args.batch_manifest is not None:
                pending = pending_batch_items(
                    args.batch_manifest,
                    args.batch_output,
                )
                if not pending:
                    print(
                        "managed-vllm: batch completed by an earlier lease; "
                        "exiting without touching a GPU",
                        flush=True,
                    )
                    append_event(
                        events_path,
                        "invocation_skipped",
                        run_id,
                        profile=args.profile,
                        reason="completed_by_earlier_lease",
                        invocation_wait_seconds=(
                            time.monotonic() - invocation_started
                        ),
                    )
                    return 0

            endpoint = str(profile["endpoint"])
            model = str(profile["model"])
            if endpoint_has_model(endpoint, model):
                raise RuntimeError(
                    f"{model} is already resident at {endpoint} outside this lease; "
                    "refusing to stop an unmanaged server"
                )

            gpu_deadline = time.monotonic() + args.gpu_wait_timeout
            selected_gpu = wait_for_gpu(profile, gpu_deadline)
            environment = os.environ.copy()
            environment.update(
                {
                    str(key): str(value)
                    for key, value in profile.get("env", {}).items()
                }
            )
            if selected_gpu is not None:
                # The allow-list is the authority. Override a legacy fixed
                # CUDA_VISIBLE_DEVICES value only for this managed child.
                environment["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)

            server_process: subprocess.Popen[bytes] | None = None
            server_started: float | None = None
            server_ready: float | None = None
            client_started: float | None = None
            client_return_code: int | None = None
            try:
                with log_path.open("ab", buffering=0) as log_handle:
                    server_process = subprocess.Popen(
                        [str(value) for value in profile["server_argv"]],
                        env=environment,
                        stdout=log_handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        preexec_fn=set_parent_death_signal,
                    )
                    server_started = time.monotonic()
                    write_state(
                        state_path,
                        args.profile,
                        server_process,
                        run_id,
                        selected_gpu,
                    )
                    append_event(
                        events_path,
                        "server_started",
                        run_id,
                        profile=args.profile,
                        model=model,
                        gpu=selected_gpu,
                        pid=server_process.pid,
                    )
                    print(
                        f"managed-vllm: started {model} as PID "
                        f"{server_process.pid}; log={log_path}",
                        flush=True,
                    )
                    startup_deadline = time.monotonic() + args.startup_timeout
                    wait_for_server(
                        endpoint,
                        model,
                        server_process,
                        startup_deadline,
                        args.startup_poll_seconds,
                    )
                    server_ready = time.monotonic()
                    append_event(
                        events_path,
                        "server_ready",
                        run_id,
                        profile=args.profile,
                        model=model,
                        gpu=selected_gpu,
                        startup_seconds=server_ready - server_started,
                    )
                    client_started = time.monotonic()
                    client_return_code = run_client(
                        args.client_argv,
                        args.shutdown_timeout,
                    )
                    append_event(
                        events_path,
                        "client_finished",
                        run_id,
                        profile=args.profile,
                        return_code=client_return_code,
                        client_seconds=time.monotonic() - client_started,
                    )
                    return client_return_code
            finally:
                if server_process is not None:
                    print(f"managed-vllm: unloading {model}", flush=True)
                    stop_process_group(server_process, args.shutdown_timeout)
                    output_after = (
                        count_jsonl_records(args.batch_output)
                        if args.batch_output is not None
                        else None
                    )
                    append_event(
                        events_path,
                        "server_unloaded",
                        run_id,
                        profile=args.profile,
                        model=model,
                        gpu=selected_gpu,
                        client_return_code=client_return_code,
                        output_records_before=output_before,
                        output_records_after=output_after,
                        output_records_added=(
                            output_after - output_before
                            if output_after is not None
                            and output_before is not None
                            else None
                        ),
                        ready_residency_seconds=(
                            time.monotonic() - server_ready
                            if server_ready is not None
                            else None
                        ),
                        total_gpu_lease_seconds=(
                            time.monotonic() - server_started
                            if server_started is not None
                            else None
                        ),
                    )
                state_path.unlink(missing_ok=True)
    except TerminationRequested as exc:
        print(f"managed-vllm: {exc}; cleanup complete", flush=True)
        append_event(
            events_path,
            "invocation_terminated",
            run_id,
            profile=args.profile,
            signal=exc.signum,
        )
        return 128 + exc.signum
    except KeyboardInterrupt:
        print("managed-vllm: interrupted; cleanup complete", flush=True)
        append_event(
            events_path,
            "invocation_terminated",
            run_id,
            profile=args.profile,
            signal="SIGINT",
        )
        return 130
    except Exception as exc:
        append_event(
            events_path,
            "invocation_failed",
            run_id,
            profile=args.profile,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise
    finally:
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    raise SystemExit(main())
