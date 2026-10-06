#!/bin/bash
# Resume deliberately delayed shadow-score shards after the active wave exits.
# This is non-destructive: each scorer resumes its own append-only output and
# also skips every success in the frozen base and prior eight-way ledgers.
set -euo pipefail

ROOT=${1:?usage: resume_full_corpus_delayed_shards.sh ROOT RUN_DIR}
RUN=${2:?usage: resume_full_corpus_delayed_shards.sh ROOT RUN_DIR}
TOTAL_SHARDS=${TOTAL_SHARDS:-12}
DELAYED_SHARDS=${DELAYED_SHARDS:-"10 11"}
PY=/lfs/skampere3/0/alexspan/envs/ai_usage/bin/python

cd "$ROOT"
while true; do
    active=$(
        ps -C python -o args= \
            | awk "/score_visual_scene_baselines.py/ && /--num-shards $TOTAL_SHARDS/" \
            | wc -l
    )
    [ "$active" -eq 0 ] && break
    sleep 60
done

mkdir -p logs/full_corpus_shadow_shards12
for pillar in instructional witnessed commentary; do
    case "$pillar" in
        instructional) base="$RUN/instructional_low_level.jsonl" ;;
        witnessed) base="$RUN/witnessed_low_level.jsonl" ;;
        commentary) base="$RUN/commentary_low_level_v2_sequential.jsonl" ;;
    esac
    completed=(--completed-from "$base")
    for old in "$RUN"/"${pillar}"_low_level_shard_*_of_08.jsonl; do
        completed+=(--completed-from "$old")
    done
    for shard in $DELAYED_SHARDS; do
        printf -v suffix "%02d" "$shard"
        out="$RUN/${pillar}_low_level_shard_${suffix}_of_12.jsonl"
        log="logs/full_corpus_shadow_shards12/${pillar}_${suffix}_resume.log"
        nohup nice -n 15 "$PY" scripts/score_visual_scene_baselines.py \
            --manifest "$RUN/${pillar}_manifest.jsonl" \
            --out "$out" \
            --frames 12 \
            --device cpu \
            --num-shards "$TOTAL_SHARDS" \
            --shard-index "$shard" \
            "${completed[@]}" \
            > "$log" 2>&1 < /dev/null &
    done
done
