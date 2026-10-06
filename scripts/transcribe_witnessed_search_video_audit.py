#!/usr/bin/env python3
"""Transcribe selected, hash-verified search-shadow artifacts outside live state."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.state import load_config  # noqa: E402
from src.transcribe import Transcriber  # noqa: E402


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ordinal", type=int, action="append", required=True)
    parser.add_argument(
        "--language",
        help="force the ASR language for audit reruns when automatic detection is demonstrably wrong",
    )
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    wanted = set(args.ordinal)
    records = [row for row in payload["records"] if row["ordinal"] in wanted]
    if {row["ordinal"] for row in records} != wanted:
        raise SystemExit("one or more selected ordinals are absent")
    if any(row.get("artifact_status") != "rendered" for row in records):
        raise SystemExit("cannot transcribe an unrendered artifact")

    transcript_dir = args.out / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["paths"]["transcripts"] = str(transcript_dir.resolve())
    cfg["transcribe"]["device_index"] = 0
    cfg["transcribe"]["diarize"] = False
    if args.language:
        cfg["transcribe"]["language"] = args.language
    transcriber = Transcriber(cfg)

    output = []
    for row in records:
        media = args.artifact_root / row["media_path"]
        if file_sha256(media) != row["media_sha256"]:
            raise SystemExit(f"media hash mismatch: {row['uid']}")
        transcript = transcriber.transcribe(media, row["uid"])
        path = transcript_dir / f"{row['uid']}.json"
        output.append({
            "ordinal": row["ordinal"],
            "uid": row["uid"],
            "media_sha256": row["media_sha256"],
            "language": transcript.get("language"),
            "segments": len(transcript.get("segments") or []),
            "transcript_path": str(path.relative_to(args.out)),
            "transcript_sha256": file_sha256(path),
        })
    (args.out / "transcript_manifest.json").write_text(
        json.dumps({
            "kind": "witnessed_search_shadow_transcripts",
            "forced_language": args.language,
            "records": output,
        },
                   indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
