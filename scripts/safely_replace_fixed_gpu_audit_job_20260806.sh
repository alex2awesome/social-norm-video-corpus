#!/usr/bin/env bash
# Stop only the verified, pre-server managed audit child and resume from frozen
# shadow artifacts with the dynamic allowed-GPU profile.

set -uo pipefail

old_pid=$1
stage=$2
bundle=$3
state_file=$4

if kill -0 "$old_pid" 2>/dev/null; then
  old_cmd=$(tr '\000' ' ' < "/proc/$old_pid/cmdline")
  case "$old_cmd" in
    *run_fresh_three_pillar_audit_batch_20260806.sh*) ;;
    *) printf 'ABORT_UNEXPECTED_PARENT=%s\n' "$old_cmd"; exit 41 ;;
  esac
  if [[ -e "$state_file" ]]; then
    printf 'ABORT_MODEL_ALREADY_STARTED\n'
    exit 42
  fi
  mapfile -t children < <(pgrep -P "$old_pid" || true)
  if [[ "${#children[@]}" -ne 1 ]]; then
    printf 'ABORT_CHILD_COUNT=%s\n' "${#children[@]}"
    exit 43
  fi
  child_pid=${children[0]}
  child_cmd=$(tr '\000' ' ' < "/proc/$child_pid/cmdline")
  case "$child_cmd" in
    *run_managed_vllm_batch.py*) ;;
    *) printf 'ABORT_UNEXPECTED_CHILD=%s\n' "$child_cmd"; exit 44 ;;
  esac
  kill -TERM "$child_pid"
  attempt=0
  while kill -0 "$child_pid" 2>/dev/null && [[ "$attempt" -lt 30 ]]; do
    sleep 1
    attempt=$((attempt + 1))
  done
  if kill -0 "$child_pid" 2>/dev/null; then
    printf 'ABORT_CHILD_DID_NOT_STOP=%s\n' "$child_pid"
    exit 45
  fi
  attempt=0
  while kill -0 "$old_pid" 2>/dev/null && [[ "$attempt" -lt 30 ]]; do
    sleep 1
    attempt=$((attempt + 1))
  done
  if kill -0 "$old_pid" 2>/dev/null; then
    printf 'ABORT_PARENT_DID_NOT_STOP=%s\n' "$old_pid"
    exit 46
  fi
fi

nohup /bin/bash "$bundle/scripts/resume_fresh_three_pillar_audit_batch_20260806.sh" \
  /lfs/skampere3/0/alexspan/norm-scraper "$stage" "$bundle" \
  > "$stage/fresh_run_v1_resume.log" 2>&1 < /dev/null &
resume_pid=$!
printf '%s\n' "$resume_pid" > "$stage/fresh_run_v1_resume.pid"
printf 'RESUME_JOB_PID=%s\nRESUME_LOG=%s\n' \
  "$resume_pid" "$stage/fresh_run_v1_resume.log"
