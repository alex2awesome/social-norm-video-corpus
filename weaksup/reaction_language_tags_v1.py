#!/usr/bin/env python3
"""Tag the language of witnessed reaction texts (append-only ledger).

Dependency-free two-stage guess: dominant Unicode script first (Cyrillic,
Arabic, Devanagari, CJK, ...), then stopword voting among common Latin-script
languages.  Tags are honest heuristics for stratifying audits and manifests —
``language_guess`` is a script- or stopword-level call, never certified.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

TAGGER_VERSION = "reaction_language_tags_v1"

SCRIPT_RANGES = (
    ("cyrillic", 0x0400, 0x04FF),
    ("greek", 0x0370, 0x03FF),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
    ("hebrew", 0x0590, 0x05FF),
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("tamil", 0x0B80, 0x0BFF),
    ("thai", 0x0E00, 0x0E7F),
    ("hangul", 0xAC00, 0xD7AF),
    ("cjk", 0x4E00, 0x9FFF),
    ("kana", 0x3040, 0x30FF),
)

LATIN_STOPWORDS = {
    "english": {"the", "you", "that", "what", "this", "is", "are", "don't",
                "not", "hell", "get", "your", "why", "stop", "of", "and"},
    "spanish": {"que", "qué", "por", "está", "estás", "eso", "los", "las",
                "una", "pero", "hombre", "haciendo", "usted", "mira"},
    "portuguese": {"que", "você", "não", "isso", "está", "fazendo", "uma",
                   "cara", "meu", "aqui", "por", "voce", "nao"},
    "french": {"que", "vous", "pas", "est", "les", "qu'est", "fais", "quoi",
               "c'est", "mais", "pourquoi", "arrête"},
    "german": {"das", "sie", "nicht", "was", "ist", "der", "die", "und",
               "machst", "warum", "aufhören", "doch"},
    "italian": {"che", "non", "cosa", "sei", "sta", "stai", "perché", "una",
                "questo", "fai", "basta"},
    "indonesian_malay": {"yang", "itu", "kamu", "tidak", "apa", "ini", "kenapa",
                         "jangan", "orang", "saja"},
    "tagalog": {"ang", "mo", "ka", "hindi", "ano", "yan", "bakit", "wag",
                "naman", "iyan"},
    "turkish": {"bir", "ne", "yapıyorsun", "değil", "sen", "bu", "niye",
                "yapma", "hayır"},
    "vietnamese": {"không", "cái", "làm", "gì", "này", "sao", "đừng", "bạn"},
}
# Words too common across languages to be decisive on their own.
AMBIGUOUS = {"que", "por", "una", "est", "les", "not", "no", "die", "der"}

MIN_LETTERS = 6


def dominant_script(text: str) -> tuple[str, float]:
    counts: dict[str, int] = {}
    letters = 0
    for char in text:
        if not char.isalpha():
            continue
        letters += 1
        code = ord(char)
        name = "latin"
        for script, low, high in SCRIPT_RANGES:
            if low <= code <= high:
                name = script
                break
        counts[name] = counts.get(name, 0) + 1
    if not letters:
        return "none", 0.0
    script = max(counts, key=counts.get)
    return script, counts[script] / letters


def latin_language_guess(text: str) -> str:
    words = {
        "".join(c for c in word if c.isalpha() or c == "'").lower()
        for word in text.split()
    }
    votes = {
        language: len((stopwords & words) - AMBIGUOUS)
        + 0.5 * len(stopwords & words & AMBIGUOUS)
        for language, stopwords in LATIN_STOPWORDS.items()
    }
    best = max(votes, key=votes.get)
    ranked = sorted(votes.values(), reverse=True)
    if ranked[0] >= 1.5 and (len(ranked) < 2 or ranked[0] > ranked[1]):
        return best
    return "unknown_latin"


def tag_text(text: str) -> dict[str, Any]:
    stripped = " ".join(text.split())
    letters = sum(c.isalpha() for c in stripped)
    if letters < MIN_LETTERS:
        return {"language_guess": "too_short", "dominant_script": "none",
                "script_fraction": 0.0}
    script, fraction = dominant_script(stripped)
    if script == "latin":
        guess = latin_language_guess(stripped)
    else:
        guess = f"script:{script}"
    return {"language_guess": guess, "dominant_script": script,
            "script_fraction": round(fraction, 3)}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True,
                        help="action window proposals (defines items to tag)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    with args.out.open("x") as out:
        cache: dict[str, dict[str, Any]] = {}
        for row in iter_jsonl(args.proposals):
            uid, clip_idx = row["uid"], row["clip_idx"]
            if uid not in cache:
                try:
                    cache[uid] = json.loads(
                        (args.root / "data" / "hits" / uid / "metadata.json").read_text()
                    )
                except (OSError, json.JSONDecodeError):
                    cache[uid] = {}
            reactions = [
                r for r in cache[uid].get("reactions") or []
                if str(r.get("clip_idx")) == str(clip_idx)
            ]
            text = " ".join(
                str(r.get("matched_text") or r.get("phrase") or "") for r in reactions
            )
            tag = tag_text(text)
            counts[tag["language_guess"]] = counts.get(tag["language_guess"], 0) + 1
            out.write(json.dumps({
                "item_id": row["item_id"], "uid": uid, "clip_idx": clip_idx,
                "evidence_tier": row.get("evidence_tier"),
                "reaction_text_sample": text[:120], **tag,
                "tagger_version": TAGGER_VERSION,
            }, sort_keys=True) + "\n")
    print(json.dumps({"tagger_version": TAGGER_VERSION,
                      "by_language": dict(sorted(counts.items(), key=lambda kv: -kv[1]))},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
