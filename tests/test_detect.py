"""Offline unit tests for the reaction detector (no GPU / network needed).

Run:  python -m tests.test_detect
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import detect_reactions, state  # noqa: E402


def _words(text: str, t0: float = 0.0, dt: float = 0.4, speaker="SPEAKER_00"):
    """Build a fake WhisperX word stream from a sentence."""
    out = []
    t = t0
    for tok in text.split():
        out.append({"word": tok, "start": round(t, 3), "end": round(t + dt, 3),
                    "score": 0.9, "speaker": speaker})
        t += dt
    return out


def _transcript(words):
    return {"video_id": "TEST", "language": "en", "segments": [], "words": words}


def run():
    cfg = state.load_config()
    det = detect_reactions.ReactionDetector(cfg)

    cases = []

    # Tier 1 literal, with punctuation in source words
    m = det.detect(_transcript(_words("Wow, that's so RUDE! you can't do that.")))
    phrases = {x["phrase"] for x in m}
    cases.append(("tier1 literal", {"that's so rude", "so rude", "you can't do that"} <= phrases, phrases))

    # Tier 3 regex norm-naming
    m = det.detect(_transcript(_words("hey we don't talk at the table young man")))
    cases.append(("tier3 regex", any(x["tier"] == 3 for x in m), [x["phrase"] for x in m]))

    # "oh my god" alone should NOT fire (needs Tier-1 within 3s)
    m = det.detect(_transcript(_words("oh my god the weather is nice today")))
    cases.append(("omg suppressed", not any("oh my god" in x["phrase"] for x in m), [x["phrase"] for x in m]))

    # "oh my god" WITH a nearby tier-1 SHOULD fire
    m = det.detect(_transcript(_words("oh my god that's so rude")))
    cases.append(("omg with anchor", any("oh my god" in x["phrase"] for x in m), [x["phrase"] for x in m]))

    # Tier 4 vocative alone suppressed; with anchor kept
    m = det.detect(_transcript(_words("sir the bus is here")))
    cases.append(("vocative suppressed", not any(x["tier"] == 4 for x in m), [x["phrase"] for x in m]))
    m = det.detect(_transcript(_words("sir that is inappropriate")))
    cases.append(("vocative with anchor", any(x["tier"] == 4 for x in m), [x["phrase"] for x in m]))

    # timestamps sane + speaker propagated
    m = det.detect(_transcript(_words("that's so rude")))
    ts_ok = bool(m) and m[0]["start"] >= 0 and m[0]["end"] > m[0]["start"] and m[0]["speaker"] == "SPEAKER_00"
    cases.append(("timestamps+speaker", ts_ok, m[0] if m else None))

    ok = True
    for name, passed, detail in cases:
        flag = "PASS" if passed else "FAIL"
        if not passed:
            ok = False
        print(f"[{flag}] {name}: {detail}")
    print("\nALL PASSED" if ok else "\nSOME FAILED")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if run() else 1)
