#!/usr/bin/env python3
"""Dependency-free lexical mechanisms for witnessed reaction proposals."""

from __future__ import annotations

import re


PATTERNS = {
    "direct": re.compile(
        r"\b(?:stop|don'?t|do not|leave .* alone|back off|cut it out|"
        r"knock it off|shut up|get (?:away|out|off)|enough|"
        r"what are you doing|you can'?t|you shouldn'?t)\b",
        re.I,
    ),
    "distract": re.compile(
        r"\b(?:come with me|let'?s go|look over here|can you help me|"
        r"what time is it|change the subject)\b",
        re.I,
    ),
    "delegate": re.compile(
        r"\b(?:call|tell|get|find|report to)\s+(?:the\s+)?"
        r"(?:police|security|manager|teacher|staff|parent|supervisor)\b",
        re.I,
    ),
    "support": re.compile(
        r"\b(?:are you (?:okay|alright)|is (?:he|she|they) (?:okay|alright)|"
        r"if (?:he|she|they) is (?:okay|alright)|"
        r"i'?m here|i'?ll help|"
        r"help (?:him|her|them)|you'?re safe|come with us)\b",
        re.I,
    ),
    "evaluation": re.compile(
        r"\b(?:not (?:okay|cool|right|acceptable)|what'?s wrong with you|"
        r"what is your problem|disgusting|rude|shame on you|"
        r"too much noise|that was wrong)\b",
        re.I,
    ),
    "warning": re.compile(
        r"\b(?:watch out|be careful|you'?ll hurt|going to hurt|dangerous)\b",
        re.I,
    ),
    "delayed_followup": re.compile(
        r"\b(?:later|afterward|afterwards|eventually|the next day|"
        r"i reported|i told .* later)\b",
        re.I,
    ),
    "self_defense": re.compile(
        r"\b(?:i didn'?t|i was just|leave me alone|don'?t touch me|"
        r"get away from me|i'?m sorry|it wasn'?t me)\b",
        re.I,
    ),
    "reported": re.compile(
        r"\b(?:he said|she said|they said|according to|reportedly|"
        r"the video shows|you can see|we see|this footage)\b",
        re.I,
    ),
}
GENERIC_AFFECT = re.compile(
    r"^(?:\W*(?:oh|wow|whoa|woah|gosh|god|ah|ha|haha|lol|"
    r"what|no|yes|yeah|okay|omg)\W*){1,5}$",
    re.I,
)


# --- v2 lexicon extension (2026-09-01) -------------------------------------
# Gap phrases exposed by the norm_event_only tier census: real censure that v1
# missed ("that's not fair", targeted insults, demands for apology/respect).
# v1 above is frozen because the audited clip-wide scan rule was evaluated on
# it; v2 is a separate, not-yet-audited lexicon for new shadow LF versions.
PATTERNS_V2_EXTRA = {
    "evaluation_extra": re.compile(
        r"\b(?:(?:that'?s |it'?s |so )?not fair|how dare you|"
        r"you (?:can'?t|cannot) (?:treat|talk to|do that to)|"
        r"who do you think you are|have you no shame|"
        r"you should be ashamed|that'?s (?:messed|f\W?\w*ed) up|"
        r"unbelievable|out of line|uncalled for)\b",
        re.I,
    ),
    "demand_repair": re.compile(
        r"\b(?:apologi[zs]e|say (?:you'?re )?sorry|"
        r"(?:show|have) some respect|watch (?:it|yourself|where you'?re going)|"
        r"mind your (?:own )?(?:business|manners))\b",
        re.I,
    ),
    "targeted_insult": re.compile(
        r"\byou(?:'?re| are)? (?:a |an |such a |nothing but a )?"
        r"(?:idiot|jerk|asshole|bitch|liar|creep|animal|pig|disgrace|"
        r"psycho|lunatic|monster)\b",
        re.I,
    ),
}
ACTIVE_V2_EXTRA = ("evaluation_extra", "demand_repair", "targeted_insult")


def intervention_features(text: str) -> dict[str, float]:
    """Return non-exclusive, enumerable intervention mechanisms."""
    normalized = " ".join(text.split())
    result = {
        f"intervention.{name}": float(bool(pattern.search(normalized)))
        for name, pattern in PATTERNS.items()
    }
    result["intervention.generic_affect_only"] = float(
        bool(GENERIC_AFFECT.fullmatch(normalized))
    )
    result["intervention.any_active"] = float(
        any(
            result[f"intervention.{name}"]
            for name in (
                "direct",
                "distract",
                "delegate",
                "support",
                "evaluation",
                "warning",
            )
        )
    )
    return result


def intervention_features_v2(text: str) -> dict[str, float]:
    """v1 features plus the gap-phrase families; adds ``any_active_v2``.

    Kept separate from the frozen v1 so the audited scan rule's behavior is
    unchanged; consumers must emit distinct v2 LF ids.
    """
    normalized = " ".join(text.split())
    result = intervention_features(text)
    for name, pattern in PATTERNS_V2_EXTRA.items():
        result[f"intervention.{name}"] = float(bool(pattern.search(normalized)))
    result["intervention.any_active_v2"] = float(
        result["intervention.any_active"]
        or any(result[f"intervention.{name}"] for name in ACTIVE_V2_EXTRA)
    )
    return result
