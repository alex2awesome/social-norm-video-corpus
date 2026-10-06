"""Offline unit tests for multilingual reaction localization (no GPU/network).

Covers the 2026-06-25 multilingual change:
  * _norm_token keeps word chars in ANY script (Unicode), not just ASCII
  * transcribe._pseudo_words gives coarse timings when a language has no
    WhisperX alignment model
  * LLMDetector._locate can still map a non-Latin quote to a time span

Run:  python -m tests.test_multilingual
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import llm_detect, transcribe  # noqa: E402


def run() -> int:
    fails = []

    def check(name, cond, *info):
        print(("PASS " if cond else "FAIL ") + name, *info)
        if not cond:
            fails.append(name)

    nt = llm_detect._norm_token
    # 1. Unicode tokens survive normalization (ASCII-only class would erase them)
    check("cyrillic kept", nt("Привет!") == "привет", nt("Привет!"))
    check("accented kept", nt("Café,") == "café", nt("Café,"))
    check("arabic kept", nt("مرحبا") == "مرحبا", nt("مرحبا"))
    check("english regress", nt("RUDE!") == "rude", nt("RUDE!"))
    check("apostrophe kept", nt("don't") == "don't", nt("don't"))

    # 2. pseudo-words spread a segment span linearly over its tokens
    seg = {"text": "это просто возмутительно поведение", "start": 10.0, "end": 14.0}
    pw = transcribe._pseudo_words(seg)
    check("pseudo n tokens", len(pw) == 4, len(pw))
    check("pseudo start", pw and pw[0]["start"] == 10.0, pw[0] if pw else None)
    check("pseudo end", pw and abs(pw[-1]["end"] - 14.0) < 1e-6, pw[-1] if pw else None)
    # space-less script -> per-character fallback so a >=2-unit quote can locate
    cjk = transcribe._pseudo_words({"text": "太过分了", "start": 0.0, "end": 2.0})
    check("cjk char split", len(cjk) == 4, len(cjk))

    # 3. a Cyrillic quote locates within the pseudo-word stream.
    # NB _locate requires >=3 tokens (loop floor range(len(q), 2, -1)), so the
    # quote must be at least three words -- a 2-word quote never matches.
    det = llm_detect.LLMDetector({"clip": {}, "llm": {}})
    wt = det._word_tokens(pw)
    loc = det._locate("просто возмутительно поведение", wt)
    check("cyrillic locate", loc is not None and abs(loc[0] - 11.0) < 1e-6, loc)

    # 4. English regression: real word stream still locates
    en_words = [{"word": w, "start": i * 0.5, "end": i * 0.5 + 0.4}
                for i, w in enumerate("hey watch it man".split())]
    loc_en = det._locate("watch it man", det._word_tokens(en_words))
    check("english locate", loc_en is not None, loc_en)

    print(f"\n{'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(run())
