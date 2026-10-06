"""End-to-end smoke test across source adapters (needs network + GPU).

For each platform: enumerate a few candidates, take the first short downloadable
one, run the FULL pipeline (download -> WhisperX -> detect), and report. Reuses a
single WhisperX model load across platforms.

Run on sk3:  ./run.sh python -m tests.test_sources_e2e
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import detect_reactions, scrape, sources, state
from src.transcribe import Transcriber

PLATFORM_TARGETS = [
    ("reddit", "PublicFreakout"),
    ("dailymotion", "karen freakout"),
    ("odysee", "public freakout"),
]


def run():
    cfg = state.load_config()
    state.ensure_dirs(cfg)
    cap = cfg["search"].get("max_duration_sec", 1800)
    tr = Transcriber(cfg)
    det = detect_reactions.ReactionDetector(cfg)

    for platform, target in PLATFORM_TARGETS:
        print(f"\n===== {platform} :: {target} =====")
        cands, cursor = sources.enumerate_candidates(platform, target, cfg)
        print(f"enumerated {len(cands)} candidates (next cursor={cursor})")
        if not cands:
            print("  !! no candidates"); continue

        # pick the first reasonably-short candidate we can download
        media = chosen = None
        for c in cands:
            if c.get("duration") and c["duration"] > min(cap, 300):
                continue  # keep the test fast: prefer < 5 min
            media = scrape.download(c, cfg)
            if media:
                chosen = c; break
        if not media:
            print("  !! could not download any candidate"); continue
        print(f"  downloaded {chosen['uid']} ({chosen.get('duration')}s) -> {media.name}")

        t = tr.transcribe(media, chosen["uid"])
        nwords = len(t.get("words", []))
        matches = det.detect(t)
        print(f"  transcript words={nwords} | reaction matches={len(matches)}")
        for m in matches[:4]:
            print(f"    [{m['tier']}] '{m['phrase']}' @ {m['start']:.1f}s :: {m.get('context','')[:60]}")
        scrape.purge_raw(chosen["uid"], cfg)  # keep test tidy

    print("\nDONE")


if __name__ == "__main__":
    run()
