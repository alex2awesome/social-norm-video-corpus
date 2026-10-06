#!/usr/bin/env bash
# Package a completed, read-only fresh audit after exactly one process check.
# The caller has already streamed this script and its Python dependencies into
# retrieval_dir. Corpus media is never mutated; only compact review derivatives
# (images plus hash-linked audio-only tracks for every pillar when present)
# enter the archive.

set -euo pipefail

stage=$1
run=$2
retrieval_dir=$3
output_dir=$4
archive=$5
job_pid=$6
v3_scores=$7
audit_python=$8
repo_root=$9

if kill -0 "$job_pid" 2>/dev/null; then
  printf '{"state":"running","pid":%s}\n' "$job_pid"
  exit 75
fi

summary="$run/resume_batch_summary.json"
if [[ ! -s "$summary" ]]; then
  printf '{"state":"finished_without_summary"}\n'
  exit 2
fi
cat "$summary"

cache_inventory="$stage/cached_audiovisual_model_inventory.json"
if [[ ! -e "$cache_inventory" ]]; then
  PYTHONPATH="$retrieval_dir" "$audit_python" \
    "$retrieval_dir/scripts/report_cached_audiovisual_models.py" \
    --cache-root /lfs/skampere3/0/shared_hf_cache \
    --cache-root /lfs/skampere3/0/alexspan/.cache/huggingface \
    --out "$cache_inventory"
fi

query_provenance="$stage/fresh_selection_query_provenance_v1.jsonl"
query_provenance_summary="$stage/fresh_selection_query_provenance_v1.summary.json"
if [[ ! -e "$query_provenance" && ! -e "$query_provenance_summary" ]]; then
  PYTHONPATH="$retrieval_dir" "$audit_python" \
    "$retrieval_dir/scripts/export_fresh_selection_query_provenance_v1.py" \
    --root "$repo_root" \
    --instructional "$stage/outputs/instructional_v4/sealed_selection.jsonl" \
    --witnessed "$stage/outputs/witnessed_v6/sealed_selection.jsonl" \
    --out "$query_provenance" \
    --summary "$query_provenance_summary"
elif [[ ! -s "$query_provenance" || ! -s "$query_provenance_summary" ]]; then
  printf '{"state":"partial_query_provenance_refused"}\n' >&2
  exit 2
fi

if [[ -e "$archive" && ! -f "$archive" ]]; then
  printf '{"state":"refused_nonfile_archive_target"}\n' >&2
  exit 2
fi

if [[ ! -e "$archive" ]]; then
  if [[ ! -e "$output_dir" ]]; then
    PYTHONPATH="$retrieval_dir" "$audit_python" \
      "$retrieval_dir/scripts/package_fresh_three_pillar_review_bundle.py" \
      --stage "$stage" \
      --run "$run" \
      --v3-scores "$v3_scores" \
      --ffmpeg /lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffmpeg \
      --ffprobe /lfs/skampere3/0/alexspan/envs/norm-scraper/bin/ffprobe \
      --out "$output_dir"
  elif [[ ! -s "$output_dir/bundle_summary.json" ]]; then
    printf '{"state":"refused_incomplete_existing_review_bundle"}\n' >&2
    exit 2
  fi
  archive_tmp="$archive.building.$$"
  trap 'rm -f -- "$archive_tmp"' EXIT
  tar -czf "$archive_tmp" -C "$stage" "$(basename "$output_dir")"
  mv -- "$archive_tmp" "$archive"
  trap - EXIT
fi
"$audit_python" - "$archive" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
digest = hashlib.sha256()
with path.open("rb") as handle:
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
print(json.dumps({
    "state": "ready",
    "archive": str(path),
    "bytes": path.stat().st_size,
    "sha256": digest.hexdigest(),
}, sort_keys=True))
PY
