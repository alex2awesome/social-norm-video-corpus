#!/usr/bin/env python3
"""Report locally cached audio-capable model candidates without networking.

The scanner reads Hugging Face cache names and config files only.  It does not
load weights, contact the Hub, reserve a GPU, or claim runtime compatibility.
Its output is an inventory for a separately audited audiovisual canary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


AUDIO_ARCHITECTURE_MARKERS = (
    "audio",
    "omni",
    "voxtral",
    "ultravox",
    "speech",
)


def repo_id_from_cache_dir(path: Path) -> str:
    name = path.name
    if not name.startswith("models--"):
        return name
    return name[len("models--"):].replace("--", "/")


def read_config(snapshot: Path) -> dict[str, Any] | None:
    path = snapshot / "config.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def config_markers(config: dict[str, Any]) -> list[str]:
    # Inspect values, not field names: serializing a missing `audio_config`
    # key would otherwise make every ordinary vision model look audio-capable.
    values = [
        config.get("architectures"),
        config.get("model_type"),
    ]
    for field in ("audio_config", "audio_token_id", "speech_config"):
        if config.get(field) is not None:
            values.append({field: config[field]})
    text = json.dumps(values, sort_keys=True).lower()
    return [marker for marker in AUDIO_ARCHITECTURE_MARKERS if marker in text]


def scan(cache_roots: list[Path]) -> dict[str, Any]:
    candidates = []
    roots_checked = []
    for root in cache_roots:
        roots_checked.append(str(root))
        hub = root / "hub" if (root / "hub").is_dir() else root
        if not hub.is_dir():
            continue
        for repository in sorted(hub.glob("models--*")):
            snapshots = repository / "snapshots"
            if not snapshots.is_dir():
                continue
            repo_id = repo_id_from_cache_dir(repository)
            name_markers = [
                marker for marker in AUDIO_ARCHITECTURE_MARKERS
                if marker in repo_id.lower()
            ]
            for snapshot in sorted(path for path in snapshots.iterdir() if path.is_dir()):
                config = read_config(snapshot)
                markers = sorted(set(name_markers + (config_markers(config) if config else [])))
                if not markers:
                    continue
                weight_files = list(snapshot.glob("*.safetensors")) + list(snapshot.glob("*.bin"))
                candidates.append({
                    "repo_id": repo_id,
                    "snapshot": str(snapshot),
                    "audio_capability_markers": markers,
                    "config_present": config is not None,
                    "weight_files_present": bool(weight_files),
                    "weight_file_count": len(weight_files),
                    "model_type": config.get("model_type") if config else None,
                    "architectures": config.get("architectures") if config else None,
                    "runtime_compatibility_validated": False,
                })
    return {
        "kind": "cached_audiovisual_model_inventory",
        "cache_roots_checked": roots_checked,
        "candidate_snapshots": candidates,
        "candidate_count": len(candidates),
        "network_accessed": False,
        "models_loaded": False,
        "gpu_reserved": False,
        "runtime_compatibility_validated": False,
        "automatic_acceptance": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    report = scan(args.cache_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
