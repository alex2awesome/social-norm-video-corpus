#!/bin/bash
set -euo pipefail

STAGE=${1:?usage: apply_typical_social_norm_v1.sh STAGE_DIR}
REPO=/lfs/skampere3/0/alexspan/norm-scraper
PY=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="$REPO/data/deployment_backups/${STAMP}_typical_social_norm_v1"
AUDIT_REL=audit_runs/20260806_query_trajectory_typical_norms_v1

verify_live_base() {
  local expected=$1 path=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [ "$actual" != "$expected" ]; then
    echo "ABORT live file changed since audited snapshot: $path expected=$expected actual=$actual" >&2
    exit 3
  fi
}

verify_live_base e7b73ee0e78d8ac1e75c33d781107ae2b28e2aa43101407db241f0460223c32c "$REPO/src/state.py"
verify_live_base 59ff701232b840c18b48e418d15cfb4f463898bbd3e1f5be8b16f8808b5f8970 "$REPO/src/search_loop.py"
verify_live_base 77d1524b75751ab06b3dbdbd514044cc0ade6afeaf836b4154938983147870a8 "$REPO/config/settings.yaml"

mkdir -p "$BACKUP/src" "$BACKUP/config" "$REPO/$AUDIT_REL"
cp "$REPO/src/state.py" "$REPO/src/search_loop.py" "$REPO/src/sources.py" "$BACKUP/src/"
cp "$REPO/config/settings.yaml" "$BACKUP/config/"

install -m 0644 "$STAGE/src/state.py" "$REPO/src/state.py"
install -m 0644 "$STAGE/src/search_loop.py" "$REPO/src/search_loop.py"
install -m 0644 "$STAGE/config/settings.yaml" "$REPO/config/settings.yaml"
install -m 0644 "$STAGE/config/typical_social_norm_queries_v1.yaml" "$REPO/config/typical_social_norm_queries_v1.yaml"
install -m 0755 "$STAGE/scripts/seed_typical_social_norm_queries_v1.py" "$REPO/scripts/seed_typical_social_norm_queries_v1.py"
install -m 0644 "$STAGE/$AUDIT_REL/manual_query_wording_audit.jsonl" "$REPO/$AUDIT_REL/manual_query_wording_audit.jsonl"
install -m 0644 "$STAGE/$AUDIT_REL/trajectory_analysis.json" "$REPO/$AUDIT_REL/trajectory_analysis.json"
install -m 0644 "$STAGE/$AUDIT_REL/scheduler_simulation_300.json" "$REPO/$AUDIT_REL/scheduler_simulation_300.json"
install -m 0644 "$STAGE/$AUDIT_REL/AUDIT_REPORT.md" "$REPO/$AUDIT_REL/AUDIT_REPORT.md"

if ! grep -q 'host_attempt_limits' "$REPO/src/sources.py"; then
  patch -d "$REPO" -p1 --forward < "$STAGE/patches/sources_host_attempt_limits.patch"
fi

cd "$REPO"
"$PY" -m py_compile src/state.py src/search_loop.py src/sources.py scripts/seed_typical_social_norm_queries_v1.py
"$PY" scripts/seed_typical_social_norm_queries_v1.py \
  --plan config/typical_social_norm_queries_v1.yaml \
  --manual-audit "$AUDIT_REL/manual_query_wording_audit.jsonl" \
  --trajectory-analysis "$AUDIT_REL/trajectory_analysis.json" \
  --settings config/settings.yaml \
  --expected-plan-sha256 b2f6fd8afc6a54b7c2533c3b3e4f0838e5a80419fcce4ec882da30e8a628e90c \
  --expected-analysis-sha256 5b14d99d8ecf9164920d3b4f11b8482f6467d358cef67f01aa5f604fdabb2898 \
  | tee "$AUDIT_REL/remote_seed_report.json"

mapfile -t SEARCH_PIDS < <(pgrep -f 'src.search_loop' || true)
MAIN_PIDS=()
for pid in "${SEARCH_PIDS[@]}"; do
  cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null || true)
  [ "$cwd" = "$REPO" ] || continue
  ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')
  child=0
  for candidate in "${SEARCH_PIDS[@]}"; do
    [ "$candidate" = "$ppid" ] && child=1
  done
  [ "$child" -eq 0 ] && MAIN_PIDS+=("$pid")
done
if [ "${#MAIN_PIDS[@]}" -ne 1 ]; then
  echo "ABORT restart: expected one MAIN crawl, found ${#MAIN_PIDS[@]}; installed files and backup preserved" >&2
  exit 4
fi

OLD_PID=${MAIN_PIDS[0]}
kill -TERM "$OLD_PID"
for _ in $(seq 1 20); do
  kill -0 "$OLD_PID" 2>/dev/null || break
  sleep 1
done
if kill -0 "$OLD_PID" 2>/dev/null; then
  echo "ABORT restart: MAIN pid $OLD_PID did not stop after TERM" >&2
  exit 5
fi
./crawl_watchdog.sh

"$PY" - <<'PY'
import json, os, sqlite3, subprocess
repo = "/lfs/skampere3/0/alexspan/norm-scraper"
conn = sqlite3.connect(repo + "/data/state.db")
conn.row_factory = sqlite3.Row
seed = [dict(r) for r in conn.execute(
    "SELECT family, active, count(*) AS n FROM queries "
    "WHERE source='typical_social_norm_v1' GROUP BY family, active ORDER BY family, active"
)]
policy = conn.execute("SELECT count(*) FROM queries WHERE policy_excluded=1").fetchone()[0]
runs_table = conn.execute(
    "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='query_runs'"
).fetchone()[0]
main = []
for raw in subprocess.check_output(["pgrep", "-f", "src.search_loop"], text=True).split():
    try:
        if os.path.realpath(f"/proc/{raw}/cwd") == repo:
            main.append(int(raw))
    except OSError:
        pass
print(json.dumps({
    "deployment": "typical_social_norm_v1",
    "seed_rows": seed,
    "policy_excluded_queries": policy,
    "query_runs_table": bool(runs_table),
    "main_repo_search_loop_pids_including_children": main,
}, indent=2, sort_keys=True))
PY

echo "backup=$BACKUP"
