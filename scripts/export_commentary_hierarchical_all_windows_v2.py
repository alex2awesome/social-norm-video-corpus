#!/usr/bin/env python3
"""Export complete source tiling before cheap commentary feature scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from scripts.build_commentary_hierarchical_window_manifest_v2 import (
        build_windows,
        read_jsonl,
        validate_full_coverage,
        write_jsonl,
    )
except ModuleNotFoundError:
    from build_commentary_hierarchical_window_manifest_v2 import (  # type: ignore[no-redef]
        build_windows,
        read_jsonl,
        validate_full_coverage,
        write_jsonl,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--window-sec", type=float, default=12)
    parser.add_argument("--stride-sec", type=float, default=8)
    args = parser.parse_args()
    for path in (args.out, args.summary):
        if path.exists():
            raise FileExistsError(path)
    windows = build_windows(
        read_jsonl(args.sources),
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
    )
    validate_full_coverage(windows)
    write_jsonl(args.out, windows)
    summary = {
        "kind": "commentary_hierarchical_complete_tiling_v2",
        "sources": len({row["uid"] for row in windows}),
        "windows": len(windows),
        "full_source_tiling_verified": True,
        "candidate_generation_only": True,
        "automatic_acceptance": False,
        "corpus_mutation_authorized": False,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
