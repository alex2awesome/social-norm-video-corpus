#!/bin/bash
set -euo pipefail

STAGE=${1:?usage: apply_dailymotion_related_quarantine_v1.sh STAGE_DIR}
REPO=/lfs/skampere3/0/alexspan/norm-scraper
PY=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="$REPO/data/deployment_backups/${STAMP}_dailymotion_related_quarantine_v1"
AUDIT=audit_runs/20260808_related_snowball_gate_v1

verify() {
  local expected=$1 path=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [ "$actual" = "$expected" ] || {
    echo "ABORT changed live file: $path expected=$expected actual=$actual" >&2
    exit 3
  }
}
verify b69d6d4db7b67e11422b509c1d9b008547792bb9d93d393c6b566d18218ac3f2 "$REPO/src/state.py"
verify b3edd699896a055be87c33b5f6ce7b11deeba7c2c9fd5e105a62ffe9aa9c9b17 "$REPO/src/search_loop.py"
verify 5b09fde58793d611dd6ad8fb0e1ba757f520bb0310617f518abc22ff9a551dad "$REPO/config/settings.yaml"

mkdir -p "$BACKUP/src" "$BACKUP/config" "$REPO/$AUDIT/transfer_audit"
cp "$REPO/src/state.py" "$REPO/src/search_loop.py" "$BACKUP/src/"
cp "$REPO/config/settings.yaml" "$BACKUP/config/"
install -m 0644 "$STAGE/src/state.py" "$REPO/src/state.py"
install -m 0644 "$STAGE/src/search_loop.py" "$REPO/src/search_loop.py"
install -m 0644 "$STAGE/src/snowball_scope.py" "$REPO/src/snowball_scope.py"
install -m 0644 "$STAGE/config/settings.yaml" "$REPO/config/settings.yaml"
install -m 0755 "$STAGE/scripts/quarantine_dailymotion_related_v1.py" "$REPO/scripts/quarantine_dailymotion_related_v1.py"
install -m 0644 "$STAGE/$AUDIT/manual_parent_audit.jsonl" "$REPO/$AUDIT/manual_parent_audit.jsonl"
install -m 0644 "$STAGE/$AUDIT/evaluation.json" "$REPO/$AUDIT/evaluation.json"
install -m 0644 "$STAGE/$AUDIT/transfer_audit/manual_child_audit.jsonl" "$REPO/$AUDIT/transfer_audit/manual_child_audit.jsonl"
install -m 0644 "$STAGE/$AUDIT/transfer_audit/evaluation.json" "$REPO/$AUDIT/transfer_audit/evaluation.json"
install -m 0644 "$STAGE/$AUDIT/transfer_audit/population_summary.json" "$REPO/$AUDIT/transfer_audit/population_summary.json"

cd "$REPO"
"$PY" -m py_compile src/state.py src/search_loop.py src/snowball_scope.py scripts/quarantine_dailymotion_related_v1.py
"$PY" - <<'PY'
from src import state
cfg = state.load_config()
assert cfg["snowball"]["enabled"] is True
assert cfg["snowball"]["dailymotion_enabled"] is False
assert cfg["snowball"]["reddit_enabled"] is True
conn = state.init_db(cfg)
assert conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='snowball_proposals'").fetchone()
PY
"$PY" scripts/quarantine_dailymotion_related_v1.py --db data/state.db --mode quarantine \
  | tee "$AUDIT/remote_quarantine_report.json"

mapfile -t SEARCH_PIDS < <(pgrep -f 'src.search_loop' || true)
MAIN_PIDS=()
for pid in "${SEARCH_PIDS[@]}"; do
  cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
  [ "$cwd" = "$REPO" ] || continue
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  child=0
  for candidate in "${SEARCH_PIDS[@]}"; do [ "$candidate" = "$ppid" ] && child=1; done
  [ "$child" -eq 0 ] && MAIN_PIDS+=("$pid")
done
[ "${#MAIN_PIDS[@]}" -eq 1 ] || {
  echo "ABORT restart: expected one MAIN crawl, found ${#MAIN_PIDS[@]}" >&2
  exit 4
}
OLD_PID=${MAIN_PIDS[0]}
kill -TERM "$OLD_PID"
for _ in $(seq 1 45); do
  kill -0 "$OLD_PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$OLD_PID" 2>/dev/null; then
  kill -KILL "$OLD_PID"
fi
./crawl_watchdog.sh

"$PY" - <<'PY'
import json, os, sqlite3, subprocess, time, yaml
repo = "/lfs/skampere3/0/alexspan/norm-scraper"
time.sleep(2)
conn = sqlite3.connect(repo + "/data/state.db")
cfg = yaml.safe_load(open(repo + "/config/settings.yaml"))
pids = []
for raw in subprocess.check_output(["pgrep", "-f", "src.search_loop"], text=True).split():
    try:
        if os.path.realpath(f"/proc/{raw}/cwd") == repo:
            pids.append(int(raw))
    except OSError:
        pass
print(json.dumps({
    "dailymotion_snowball_enabled": cfg["snowball"]["dailymotion_enabled"],
    "reddit_snowball_enabled": cfg["snowball"]["reddit_enabled"],
    "eligible_dmrelated_queries": conn.execute(
        "SELECT count(*) FROM queries WHERE platform='dmrelated' AND source='related' "
        "AND active=1 AND COALESCE(policy_excluded,0)=0").fetchone()[0],
    "quarantined_dmrelated_queries": conn.execute(
        "SELECT count(*) FROM queries WHERE policy_reason='related_transfer_failed:20260808_v1'").fetchone()[0],
    "main_repo_search_loop_pids_including_children": pids,
    "pending_transcribe": conn.execute(
        "SELECT count(*) FROM seen_videos WHERE status='pending_transcribe'").fetchone()[0],
    "pending_detect": conn.execute(
        "SELECT count(*) FROM seen_videos WHERE status='pending_detect'").fetchone()[0],
}, indent=2, sort_keys=True))
PY
echo "backup=$BACKUP"
