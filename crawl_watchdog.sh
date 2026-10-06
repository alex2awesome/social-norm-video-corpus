#!/bin/bash
# Main-crawl watchdog -- relaunch `src.search_loop` if the MAIN crawl is down.
#
# Added 2026-06-14 after a 3-day SILENT collection stall: the crawl is a bare
# nohup process with nothing supervising it, and the 10k-witnessed goal is a
# multi-week unattended run. If the main crawl dies at 3am, collection just
# stops with no signal. This restarts it.
#
# Safety (the 2-crawl incident is the failure mode to avoid):
#   * flock -> only one watchdog ever acts at a time.
#   * Distinguish MAIN from the QUIET crawl by CWD, not `pgrep -f` (which
#     matches both -- they run the same module from different repo dirs).
#   * Debounce: re-check after a few seconds before launching, so a manual
#     restart's brief no-process gap doesn't trigger a duplicate.
#   * Never auto-kills; if it ever sees >1 main crawl it logs LOUDLY and bails
#     so a human resolves it (killing the wrong one is worse than a dup).
set -u
REPO=/lfs/skampere3/0/alexspan/norm-scraper
LOCK=/tmp/crawl_watchdog.lock
LOG=$REPO/logs/crawl_watchdog.log
exec 9>"$LOCK"
flock -n 9 || exit 0
stamp(){ date "+%Y-%m-%d %H:%M:%S"; }

# Count TOP-LEVEL MAIN-repo crawls. A match counts only if (a) its cwd is the
# main repo and (b) its parent is NOT itself a search_loop match -- i.e. it's
# the detached crawl, not a multiprocessing/worker fork (those share the cmdline
# and cwd and would otherwise inflate the count into a false "2 crawls" bail).
count_main(){
  local n=0 pid cwd ppid
  local pids; pids=$(pgrep -f "src.search_loop" 2>/dev/null)
  for pid in $pids; do
    cwd=$(readlink -f "/proc/$pid/cwd" 2>/dev/null)
    [ "$cwd" = "$REPO" ] || continue
    ppid=$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d " ")
    echo "$pids" | grep -qw "$ppid" && continue   # parent is a search_loop -> worker fork
    n=$((n+1))
  done
  echo "$n"
}

n=$(count_main)
if [ "$n" -ge 2 ]; then
  echo "$(stamp) WARNING: $n MAIN crawls running -- NOT acting, human needed" >> "$LOG"
  exit 1
fi
[ "$n" -eq 1 ] && exit 0   # healthy

# Debounce: a manual restart may have a brief gap. Wait and re-check.
sleep 4
n=$(count_main)
[ "$n" -ge 1 ] && exit 0   # came back (or manual restart finished)

echo "$(stamp) MAIN crawl DOWN -> relaunching" >> "$LOG"
cd "$REPO" || { echo "$(stamp) cd $REPO failed" >> "$LOG"; exit 1; }
nohup ./run.sh python -u -m src.search_loop >> "logs/run_$(date +%F).log" 2>&1 </dev/null &
NEW=$!
sleep 6
cwd=$(readlink -f "/proc/$NEW/cwd" 2>/dev/null)
if [ "$cwd" = "$REPO" ]; then
  echo "$(stamp) relaunched OK pid=$NEW" >> "$LOG"
else
  echo "$(stamp) relaunch UNCONFIRMED pid=$NEW cwd=$cwd (check run.sh/conda)" >> "$LOG"
fi
exit 0
