"""Match reaction phrases against a WhisperX word stream.

Approach
--------
1. Normalize each word to a token keeping [a-z0-9'] (so "Rude!" -> "rude",
   "don't" -> "don't"), dropping tokens that become empty. Record the char span
   of every token inside a single normalized string.
2. Literal phrases are normalized the same way and compiled to regexes with
   word boundaries; regex patterns are applied directly to the normalized text.
3. A regex match's char span is mapped back to the covering words, giving
   precise start/end times and a speaker (from the first covered word).
4. Conditional groups (Tier-4 vocatives, lone "oh my god") are kept only when a
   qualifying anchor match (an unconditional group in `with_tiers`) lies within
   the configured time window.

Returns a list of match dicts:
    {phrase, tier, tag, start, end, speaker, context}
sorted by start time.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

from . import state

log = logging.getLogger("detect")

_KEEP = re.compile(r"[^a-z0-9']+")


def _norm_token(word: str) -> str:
    return _KEEP.sub("", word.lower())


class _Pattern:
    __slots__ = ("tier", "tag", "regex", "display", "cooccur", "literal")

    def __init__(self, tier: int, tag: Optional[str], regex: re.Pattern,
                 display: str, cooccur: Optional[dict], literal: bool = True):
        self.tier = tier
        self.tag = tag
        self.regex = regex
        self.display = display
        self.cooccur = cooccur  # {window, with_tiers} or None
        self.literal = literal  # False -> `display` is a regex source, not a phrase


class ReactionDetector:
    def __init__(self, cfg: dict, phrases: Optional[dict] = None):
        self.cfg = cfg
        self.context_window = cfg["clip"].get("context_window_sec", 5.0)
        spec = phrases or state.load_yaml("config/reaction_phrases.yaml")
        self.patterns: list[_Pattern] = []
        for group in spec.get("patterns", []):
            tier = int(group["tier"])
            tag = group.get("tag")
            cooccur = group.get("cooccur")
            ptype = group.get("type", "literal")
            for item in group.get("items", []):
                if ptype == "literal":
                    norm = " ".join(_norm_token(w) for w in item.split())
                    norm = re.sub(r"\s+", " ", norm).strip()
                    if not norm:
                        continue
                    rx = re.compile(r"\b" + re.escape(norm) + r"\b")
                else:  # regex
                    rx = re.compile(item)
                self.patterns.append(_Pattern(tier, tag, rx, item, cooccur,
                                              literal=(ptype == "literal")))

    # -- text building -------------------------------------------------------
    @staticmethod
    def _build_text(words: list[dict]):
        """Return (normalized_text, spans) where spans[i]=(char_start,char_end)
        for the i-th *kept* word, plus the parallel list of kept words."""
        tokens, spans, kept = [], [], []
        pos = 0
        for w in words:
            tok = _norm_token(w.get("word", ""))
            if not tok:
                continue
            if tokens:
                pos += 1  # the joining space
            start = pos
            pos += len(tok)
            tokens.append(tok)
            spans.append((start, pos))
            kept.append(w)
        return " ".join(tokens), spans, kept

    @staticmethod
    def _covering_words(spans, lo, hi):
        """Indices of words whose char span intersects [lo, hi)."""
        idx = []
        for i, (a, b) in enumerate(spans):
            if a < hi and b > lo:
                idx.append(i)
        return idx

    def _context(self, kept, t_start, t_end) -> str:
        win = self.context_window
        ctx = [w["word"].strip() for w in kept
               if w["end"] >= t_start - win and w["start"] <= t_end + win]
        return " ".join(ctx).strip()

    # -- main ----------------------------------------------------------------
    def detect(self, transcript: dict) -> list[dict]:
        words = transcript.get("words", [])
        if not words:
            return []
        text, spans, kept = self._build_text(words)

        raw: list[dict] = []
        for pat in self.patterns:
            for m in pat.regex.finditer(text):
                cov = self._covering_words(spans, m.start(), m.end())
                if not cov:
                    continue
                w0, w1 = kept[cov[0]], kept[cov[-1]]
                raw.append({
                    # literal patterns keep their canonical phrase; regex patterns
                    # have no human phrase, so store the actual matched words (not
                    # the regex source, which used to leak in as garbage).
                    "phrase": pat.display if pat.literal else m.group(0),
                    "matched_text": m.group(0),
                    "tier": pat.tier,
                    "tag": pat.tag,
                    "start": w0["start"],
                    "end": w1["end"],
                    "speaker": w0.get("speaker"),
                    "context": self._context(kept, w0["start"], w1["end"]),
                    "_cooccur": pat.cooccur,
                })

        kept_matches = self._apply_cooccurrence(raw)
        kept_matches.sort(key=lambda d: d["start"])
        for d in kept_matches:
            d.pop("_cooccur", None)
        return kept_matches

    def _apply_cooccurrence(self, raw: list[dict]) -> list[dict]:
        # anchors = unconditional matches usable to satisfy a cooccur requirement
        anchors = [m for m in raw if m["_cooccur"] is None]
        out = []
        for m in raw:
            req = m["_cooccur"]
            if req is None:
                out.append(m)
                continue
            window = req.get("window", 5.0)
            with_tiers = set(req.get("with_tiers", []))
            ok = any(
                a is not m
                and a["tier"] in with_tiers
                and abs(a["start"] - m["start"]) <= window
                for a in anchors
            )
            if ok:
                out.append(m)
        return out


def detect_file(transcript_path: str, cfg: Optional[dict] = None) -> list[dict]:
    """Convenience: run detection on a saved transcript JSON."""
    import json
    cfg = cfg or state.load_config()
    with open(transcript_path) as f:
        t = json.load(f)
    return ReactionDetector(cfg).detect(t)
