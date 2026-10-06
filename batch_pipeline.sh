#!/bin/bash
# GPU-free-crawl batch pipeline (llm.deferred). Cron'd hourly; flock'd so runs
# never overlap. Exits fast when the queue is small AND fresh. Each GPU stage
# WAITS (polls nvidia-smi every 2 min, up to batch.gpu_wait_minutes) until ANY
# GPU has enough free memory, then releases it completely on process exit.
#   stage 1: src.batch_transcribe  (WhisperX, norm-scraper env, ~14 GB)
#   stage 2: src.batch_detect      (offline vLLM 70B-FP8, ai_usage env, ~100 GB)
#   stage 3: src.batch_audio_events (PANNs SED over NEW hits, ~3 GB, skips scored)
export HOME=/lfs/skampere3/0/alexspan
ROOT=/lfs/skampere3/0/alexspan/norm-scraper
NS_PY=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin/python
AI_PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python
cd "$ROOT" || exit 1

# GPUs we must NEVER use on this node (reserved by other work / hardware issues).
# pick_gpu skips these indices entirely. Space-separated. Edit freely.
GPU_BLOCKLIST="1 2 3 4"

exec 9>"$ROOT/.batch_pipeline.lock"
flock -n 9 || { echo "$(date '+%F %T') another run holds the lock; exiting"; exit 0; }

# ---- thresholds + queue state (single python read of config + db) -----------
read -r N_TS N_DET AGE_H MIN_BACKLOG MAX_AGE_H MINGB_W MINGB_V WAIT_MIN <<EOF
$($NS_PY - <<'PY'
import sqlite3, time, yaml
cfg = yaml.safe_load(open("config/settings.yaml"))
b = cfg.get("batch", {})
c = sqlite3.connect("data/state.db"); c.execute("pragma busy_timeout=60000")
n1 = c.execute("select count(*) from seen_videos where status='pending_transcribe'").fetchone()[0]
n2 = c.execute("select count(*) from seen_videos where status='pending_detect'").fetchone()[0]
old = c.execute("select min(enumerated_at) from seen_videos where status in ('pending_transcribe','pending_detect')").fetchone()[0]
age = (time.time() - old) / 3600 if old else 0
print(n1, n2, f"{age:.1f}", b.get("min_backlog", 150), b.get("max_age_hours", 4),
      b.get("min_free_gb_whisper", 14), b.get("min_free_gb_vllm", 100),
      b.get("gpu_wait_minutes", 50))
PY
)
EOF
TOTAL=$((N_TS + N_DET))
echo "$(date '+%F %T') queue: transcribe=$N_TS detect=$N_DET oldest=${AGE_H}h (run if >=$MIN_BACKLOG or older than ${MAX_AGE_H}h)"
STALE=$(awk -v a="$AGE_H" -v m="$MAX_AGE_H" 'BEGIN{print (a>=m) ? 1 : 0}')
if [ "$TOTAL" -lt "$MIN_BACKLOG" ] && [ "$STALE" -eq 0 ]; then
    echo "$(date '+%F %T') below threshold; exiting (no GPU touched)"
    exit 0
fi

# ---- wait for a big-enough GPU (any index) -----------------------------------
pick_gpu () {  # $1 = min free GB; echoes the chosen index, polls up to WAIT_MIN
    local need_mb=$(( $1 * 1024 )) waited=0 best bestfree
    while true; do
        best=-1; bestfree=0
        while IFS=, read -r idx free; do
            free=${free// /}
            case " $GPU_BLOCKLIST " in *" $idx "*) continue;; esac   # reserved -> skip
            if [ "$free" -ge "$need_mb" ] && [ "$free" -gt "$bestfree" ]; then
                best=$idx; bestfree=$free
            fi
        done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
        if [ "$best" -ge 0 ]; then echo "$best"; return 0; fi
        waited=$(( waited + 2 ))
        if [ "$waited" -ge "$WAIT_MIN" ]; then return 1; fi
        echo "$(date '+%F %T') no GPU with ${1}GB free; waiting (${waited}/${WAIT_MIN} min)" >&2
        sleep 120
    done
}

# ---- stage 1: transcription ---------------------------------------------------
if [ "$N_TS" -gt 0 ]; then
    GPU=$(pick_gpu "$MINGB_W") || { echo "$(date '+%F %T') no GPU for WhisperX within ${WAIT_MIN}min; retry next cron"; exit 0; }
    echo "$(date '+%F %T') stage 1 (transcribe) on GPU $GPU"
    if ! CUDA_VISIBLE_DEVICES=$GPU "$ROOT/run.sh" python -u -m src.batch_transcribe; then
        echo "$(date '+%F %T') ERROR stage 1 (transcribe) failed; stopping pipeline" >&2
        exit 1
    fi
fi

# ---- stage 2: detection -------------------------------------------------------
N_DET2=$($NS_PY -c "import sqlite3; c=sqlite3.connect('data/state.db'); c.execute('pragma busy_timeout=60000'); print(c.execute(\"select count(*) from seen_videos where status='pending_detect'\").fetchone()[0])")
if [ "$N_DET2" -gt 0 ]; then
    GPU=$(pick_gpu "$MINGB_V") || { echo "$(date '+%F %T') no GPU for vLLM within ${WAIT_MIN}min; retry next cron"; exit 0; }
    echo "$(date '+%F %T') stage 2 (detect) on GPU $GPU ($N_DET2 transcripts)"
    export CUDA_VISIBLE_DEVICES=$GPU
    export PYTHONPATH=$ROOT
    export PATH=/lfs/skampere3/0/alexspan/envs/norm-scraper/bin:$PATH   # ffmpeg for clip cutting
    if ! "$AI_PY" -u -m src.batch_detect; then
        echo "$(date '+%F %T') ERROR stage 2 (detect) failed; stopping pipeline" >&2
        exit 1
    fi
fi

# ---- stage 3: audio-event scoring of new hits (Pass B) ------------------------
# batch_audio_events skips already-scored hits, so this only touches whatever
# stage 2 just created. Tiny model (~3 GB) -> reuse the WhisperX gate.
N_HITS=$(ls "$ROOT/data/hits" 2>/dev/null | wc -l)
N_SCORED=$(ls "$ROOT/data/audio_events" 2>/dev/null | wc -l)
if [ "$N_HITS" -gt "$N_SCORED" ]; then
    GPU=$(pick_gpu "$MINGB_W") || { echo "$(date '+%F %T') no GPU for audio events; retry next cron"; exit 0; }
    echo "$(date '+%F %T') stage 3 (audio events) on GPU $GPU ($((N_HITS - N_SCORED)) new hits)"
    if ! CUDA_VISIBLE_DEVICES=$GPU "$ROOT/run.sh" python -u -m src.batch_audio_events; then
        echo "$(date '+%F %T') ERROR stage 3 (audio events) failed; stopping pipeline" >&2
        exit 1
    fi
fi
echo "$(date '+%F %T') pipeline done"
