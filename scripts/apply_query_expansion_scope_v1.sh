#!/bin/bash
set -euo pipefail

STAGE=${1:?usage: apply_query_expansion_scope_v1.sh STAGE_DIR}
REPO=/lfs/skampere3/0/alexspan/norm-scraper
PY=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python
STAMP=$(date +%Y%m%d_%H%M%S)
BACKUP="$REPO/data/deployment_backups/${STAMP}_query_expansion_scope_v1"

verify() {
  local expected=$1 path=$2 actual
  actual=$(sha256sum "$path" | awk '{print $1}')
  [ "$actual" = "$expected" ] || {
    echo "ABORT changed live file: $path expected=$expected actual=$actual" >&2
    exit 3
  }
}
verify 30569092c3e1d129acd4eb581d50ec3291015971af84e5aa43b972f398809497 "$REPO/src/state.py"
verify 82b187455f8ec1cc1cfdac931067c8233df65b5382996cc0ed590b5c330a69b3 "$REPO/src/search_loop.py"
verify a04c32f577cd75c00a962ad82091dffcc0a25d59cc233afb9b10671f3075f3f3 "$REPO/src/query_generator.py"

mkdir -p "$BACKUP/src" "$REPO/audit_runs/20260807_query_expansion_scope_v1"
cp "$REPO/src/state.py" "$REPO/src/search_loop.py" "$REPO/src/query_generator.py" "$BACKUP/src/"
install -m 0644 "$STAGE/src/state.py" "$REPO/src/state.py"
install -m 0644 "$STAGE/src/search_loop.py" "$REPO/src/search_loop.py"
install -m 0644 "$STAGE/src/query_generator.py" "$REPO/src/query_generator.py"
install -m 0644 "$STAGE/src/query_scope.py" "$REPO/src/query_scope.py"
install -m 0644 "$STAGE/audit_runs/20260807_query_expansion_scope_v1/evaluation.json" \
  "$REPO/audit_runs/20260807_query_expansion_scope_v1/evaluation.json"
install -m 0644 "$STAGE/audit_runs/20260807_query_expansion_scope_v1/manual_negative_scope_audit.jsonl" \
  "$REPO/audit_runs/20260807_query_expansion_scope_v1/manual_negative_scope_audit.jsonl"

cd "$REPO"
"$PY" -m py_compile src/state.py src/search_loop.py src/query_generator.py src/query_scope.py
"$PY" - <<'PY'
import json
from src import state
conn = state.init_db()
rows = conn.execute(
    "SELECT policy_reason,count(*) n FROM queries "
    "WHERE source IN ('llm_expand','exploit','explore') AND policy_excluded=1 "
    "GROUP BY policy_reason ORDER BY n DESC"
).fetchall()
print(json.dumps({"auto_scope_excluded_existing": [dict(r) for r in rows],
                  "query_proposals_table": bool(conn.execute(
                      "SELECT 1 FROM sqlite_master WHERE type='table' AND name='query_proposals'"
                  ).fetchone())}, sort_keys=True))
PY

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
for _ in $(seq 1 20); do
  kill -0 "$OLD_PID" 2>/dev/null || break
  sleep 1
done
kill -0 "$OLD_PID" 2>/dev/null && {
  echo "ABORT restart: MAIN $OLD_PID did not stop" >&2
  exit 5
}
./crawl_watchdog.sh
echo "backup=$BACKUP"
