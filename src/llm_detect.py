"""LLM-as-detector: find audible bystander reactions to social-norm violations
in a transcript, using a self-hosted Llama-3.3-70B (vLLM, OpenAI-compatible).

The LLM returns verbatim quotes; we map each quote back to the WhisperX
word-level timestamps to get the EXACT reaction time, then the loop clips
[t_react - pre, t_react + post]. The LLM also returns a verdict
(is_target / hard_negative_type) so the loop can drop staged pranks, scripted
drama, spectacles, sports, lyrics -- the pollution keyword matching can't see.

Multilingual: reactions in any language are detected and located.

Returns a dict: {is_target, hard_negative_type, matches: [match,...]}
where each match is clip_extract / state.record_reactions-compatible:
    {phrase, matched_text, tier(=5), tag(=signal), norm, start, end, speaker, context}
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

import requests

from . import state

log = logging.getLogger("llm_detect")

LLM_TIER = 5  # marks an LLM-detected reaction in the reactions table

# Strip punctuation/whitespace but KEEP word chars in ANY script (Unicode-aware),
# so non-Latin reactions (Cyrillic/Arabic/Hindi/...) and accented Latin survive
# normalization. An ASCII-only class would erase non-Latin tokens to "" and make
# every such quote unlocatable (-> the clip is dropped as a miss). \w is Unicode
# by default for str patterns; '’ is the curly apostrophe ASR/LLMs emit.
_KEEP = re.compile(r"[^\w'’]+", re.UNICODE)

SYS = """You analyze a video transcript (ASR, may be messy, ANY language) for SOCIAL-NORM VIOLATIONS and the reactions to them.

STEP 1 -- is_target: "yes" if the transcript involves a social-norm violation (someone rude/entitled/inconsiderate/aggressive/cutting in line/littering/invading space/harassing/breaking an etiquette or moral norm) AND there is an audible reaction to it OR substantive discussion of it. Otherwise "no". INCLUDE mild/quiet objections, any language.
ALSO "yes": audible bystander reactions to an AUTONOMOUS VEHICLE's or ROBOT's disruptive/unexpected behavior (a driverless car / Waymo / robotaxi blocking traffic, stuck in an intersection, going the wrong way, near-missing a pedestrian, or people honking at / confronting / coning / filming it) -- treat the vehicle/robot as the AGENT whose conduct draws the reaction.

AGENT: classify the AGENT whose conduct draws the reaction -- "autonomous_vehicle" (driverless car / Waymo / robotaxi / delivery robot / drone), "animal" (dog, alligator, wildlife, etc.), "human", or "other" (some other non-human thing/machine). Waymo / self-driving / robotaxi interactions MUST be "autonomous_vehicle".

STEP 2 -- if is_target="yes", set MODALITY:
- "witnessed": the recording captures a REAL EVENT as it happens -- a person PRESENT at the scene (bystander/victim/participant) audibly reacts to / calls out / sanctions another person's conduct. The violation occurs IN the footage.
- "commentary": the speaker is NOT at a live event but is DISCUSSING / narrating / opining about / reacting-to a norm violation -- a talking-head, podcaster, vlogger, news anchor, pundit, or reaction-video creator monologuing to camera. The norm is the TOPIC, not a witnessed in-situ moment. (Clue: one dominant speaker addressing the audience, no second on-scene party reacting.)

STEP 2b -- SCENE STRUCTURE (always, when is_target="yes"). The transcript is the ONLY evidence, so judge from what is said + the ASR-timing hint (count of speech pauses):
- scene_type: "action" = a LIVE in-situ event -- turn-taking, interruptions, several distinct voices, real-time second-person confrontation ("hey!", "stop", "get away from"), short overlapping bursts (many pauses); "narration" = ONE dominant speaker monologuing / describing / explaining throughout, few real breaks, no on-scene back-and-forth (one long continuous run, few pauses); "mixed" = both. A pure "narration" scene with a single speaker is almost always COMMENTARY (or a false positive), NOT a witnessed in-situ reaction -- weigh this heavily in STEP 2.
- n_people: best-guess integer count of distinct people present/speaking in the scene (1 = a lone narrator).
- violator_role: who commits the violation -- "subject" (a person being filmed), "camera_person" (the recorder themselves, e.g. a prankster/instigator), "third_party", or "contested" (disputed/ambiguous, e.g. an activist filming someone they accuse).
- reactor_role: who audibly reacts -- "bystander" (independent onlooker), "camera_person" (the recorder narrating/objecting), "victim", or "mixed".
- severity: 1-5, how flagrant/serious the norm violation is (1 = trivial faux pas / mild rudeness, 3 = clear public violation drawing objection, 5 = egregious -- assault, open harassment, dangerous conduct).
- reaction_strength: 1-5, how strong/visceral the audible reaction is (1 = mild remark like "excuse me", 3 = clear verbal confrontation, 5 = screaming/gasps/crowd commotion).

STEP 3 -- hard_negative_type (and is_target="no") for: SCRIPTED ACTED drama (Dhar Mann-style morality skits, actors reading on-the-nose dialogue) -> "staged_prank" -- BUT a REAL hidden-camera prank on unsuspecting members of the public IS a valid norm violation (the prankster is the violator, the genuine reactions of real strangers are the signal): keep those as is_target="yes", violator_role="camera_person"; a FULL FEATURE FILM / movie upload / TV episode / trailer / soap opera (continuous scripted dialogue, music score, story arcs) -> "scripted_media"; VIDEO-GAME gameplay / roleplay streams / let's-plays (GTA, FiveM, Minecraft, streamer commentary over a game) -> "scripted_media"; spectacle/accident/disaster (fireworks, crashes, stunts) -> "spectacle_or_accident"; sports broadcast AND scripted sports entertainment (pro WRESTLING: WWE/AEW/SmackDown -- choreographed, not a real norm violation) -> "sports_broadcast"; song lyrics -> "song_lyrics"; police bodycam/procedural with no bystander sanctioning -> "police_procedural"; NEWS broadcast / news anchor / field reporter / police press-briefing narrating an incident (a talking head, often with lower-third captions, no live on-scene bystander reaction) -> "police_procedural"; no norm violation at all -> "other".
News-headline-style titles ("man in critical condition after...", "officer fatally shoots...", "X died/injured in accident", "police investigate...") signal a NEWS report -> drop unless the footage is clearly raw candid bystander video.
You are also given the VIDEO TITLE as context. A MORALIZING / SCRIPTED-style title ("... gets humbled / taught a lesson / owned / exposed in front of everyone", heavy emoji, "you won't believe", "gone wrong") together with on-the-nose acted dialogue is STAGED drama -> "staged_prank". Do NOT drop a plausibly-candid recording on title alone, but weigh it.

Profanity or excitement ALONE is not a reaction. For "witnessed" there must be an on-scene person reacting to another's conduct; for "commentary" there must be a clear norm violation being discussed.
Output ONLY a JSON object."""


def _norm_token(w: str) -> str:
    # casefold (not lower) for correct Unicode folding across scripts
    return _KEEP.sub("", w.casefold())


class LLMDetector:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lc = cfg.get("llm", {})
        self.endpoint = self.lc.get("endpoint", "http://127.0.0.1:8001/v1/chat/completions")
        self.model = self.lc.get("model", "llama70b")
        self.timeout = self.lc.get("timeout", 150)
        self.max_tokens = self.lc.get("max_tokens", 900)
        self.max_chars = self.lc.get("max_chars", 8000)
        self.ctx_win = cfg["clip"].get("context_window_sec", 5.0)

    # -- prompt construction (shared by live HTTP + offline batch paths) -----
    def _user_prompt(self, text: str, title: Optional[str] = None,
                     timing_hint: Optional[str] = None) -> str:
        title_line = f'Video title: "{title}"\n\n' if title else ""
        hint_line = f'{timing_hint}\n\n' if timing_hint else ""
        return (
            f'{title_line}{hint_line}Transcript:\n"""{text[:self.max_chars]}"""\n\n'
            'Return a JSON object:\n'
            '{"is_target":"yes"|"no",'
            '"hard_negative_type":"none"|"staged_prank"|"scripted_media"|"spectacle_or_accident"|"sports_broadcast"|"song_lyrics"|"police_procedural"|"other",'
            '"modality":"witnessed"|"commentary",'
            '"agent":"human"|"autonomous_vehicle"|"animal"|"other",'
            '"scene_type":"action"|"narration"|"mixed",'
            '"n_people":<integer best guess of distinct people in the scene>,'
            '"violator_role":"subject"|"camera_person"|"third_party"|"contested",'
            '"reactor_role":"bystander"|"camera_person"|"victim"|"mixed",'
            '"severity":<1-5 how flagrant the violation is>,'
            '"reaction_strength":<1-5 how strong/visceral the audible reaction is>,'
            '"expand_queries":["<0-3 NEW short video-search phrases that would find MORE candid '
            'videos of THIS kind of norm violation; specific, not generic; omit if is_target=no>"],'
            '"reactions":[{"quote":"<exact verbatim substring copied from the transcript>","norm":"<short>","signal":"<type>"}]}\n'
            'modality and agent are REQUIRED when is_target="yes" (agent="non_human" for '
            'autonomous-vehicle / robot interactions). For "witnessed", quote the on-scene '
            'reactions; for "commentary", quote the speaker\'s key normative statements. '
            'List at most 20. Each quote MUST be copied EXACTLY from the transcript (so it '
            'can be located). Keep quotes short (<= 15 words). JSON only.')

    # -- live HTTP path (vLLM server); the batch path uses prepare()/finish() --
    def _post(self, messages: list[dict]) -> Optional[dict]:
        try:
            r = requests.post(self.endpoint, json={
                "model": self.model,
                "messages": messages,
                "temperature": 0, "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }, timeout=self.timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return self._parse(content)
        except Exception as e:
            log.warning("LLM call failed: %s", str(e)[:160])
            return None

    @staticmethod
    def _parse(content: str) -> dict:
        s = content.strip()
        if s.startswith("```"):
            s = re.sub(r"^```(json)?", "", s).rsplit("```", 1)[0]
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return LLMDetector._salvage(s)  # tolerate truncated/malformed JSON

    @staticmethod
    def _salvage(s: str) -> dict:
        """Recover a usable result from truncated/malformed LLM JSON: pull
        is_target / hard_negative_type via regex and every COMPLETE reaction
        object that made it through before truncation."""
        out = {"is_target": "no", "hard_negative_type": "none",
               "modality": "witnessed", "agent": "human", "reactions": []}
        m = re.search(r'"is_target"\s*:\s*"(\w+)"', s)
        if m:
            out["is_target"] = m.group(1)
        m = re.search(r'"hard_negative_type"\s*:\s*"([\w_]+)"', s)
        if m:
            out["hard_negative_type"] = m.group(1)
        m = re.search(r'"modality"\s*:\s*"(\w+)"', s)
        if m:
            out["modality"] = m.group(1)
        m = re.search(r'"agent"\s*:\s*"(\w+)"', s)
        if m:
            out["agent"] = m.group(1)
        ra = s.find('"reactions"')
        seg = s[ra:] if ra >= 0 else s
        depth, start = 0, None
        for i, ch in enumerate(seg):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    try:
                        obj = json.loads(seg[start:i + 1])
                        if isinstance(obj, dict) and "quote" in obj:
                            out["reactions"].append(obj)
                    except json.JSONDecodeError:
                        pass
                    start = None
        return out

    # -- quote -> WhisperX timestamp ----------------------------------------
    @staticmethod
    def _word_tokens(words: list[dict]):
        toks = []
        for w in words:
            t = _norm_token(w.get("word", ""))
            if not t or w.get("start") is None:
                continue
            toks.append((t, float(w["start"]), float(w["end"]), w.get("speaker")))
        return toks

    def _locate(self, quote: str, wt: list):
        """Find a quote's token sequence in the word stream; return
        (start, end, speaker) of the matched span, or None. Falls back to the
        longest matching prefix (>=3 tokens) to tolerate ASR/LLM drift."""
        q = [_norm_token(x) for x in quote.split()]
        q = [x for x in q if x]
        if len(q) < 2:
            return None
        toks = [t[0] for t in wt]
        for L in range(len(q), 2, -1):   # try full quote, then shorter prefixes
            sub = q[:L]
            for i in range(0, len(toks) - L + 1):
                if toks[i:i + L] == sub:
                    return wt[i][1], wt[i + L - 1][2], wt[i][3]
        return None

    def _context(self, wt, t0, t1) -> str:
        win = self.ctx_win
        return " ".join(t[0] for t in wt if t[2] >= t0 - win and t[1] <= t1 + win)

    # -- main ----------------------------------------------------------------
    def prepare(self, transcript: dict, title: Optional[str] = None):
        """Pre-LLM half: build the chat messages for this transcript.

        Returns (messages, aux) where aux carries what finish() needs, or
        (None, result) when no LLM call is required (transcript too short).
        Shared by the live HTTP path (detect) and the offline batch engine."""
        words = transcript.get("words", [])
        text = " ".join(w.get("word", "").strip() for w in words).strip()
        if len(text.split()) < 6:
            return None, {"is_target": "no", "hard_negative_type": "other", "matches": []}

        # ASR-timing hint: count of speech pauses helps the LLM tell continuous
        # NARRATION (one speaker, few breaks) from in-situ ACTION (turn-taking).
        spans = [(w.get("start"), w.get("end")) for w in words
                 if w.get("start") is not None and w.get("end") is not None]
        n_pause = sum(1 for (a0, a1), (b0, b1) in zip(spans, spans[1:]) if b0 - a1 > 1.5)
        dur = (spans[-1][1] - spans[0][0]) if spans else 0
        _brk = "many breaks -> likely action" if n_pause >= 6 else "few breaks -> likely continuous narration"
        timing_hint = (f"ASR timing: ~{dur:.0f}s audio, {len(words)} words, "
                       f"{n_pause} speech pauses >1.5s ({_brk}).")
        messages = [
            {"role": "system", "content": SYS},
            {"role": "user", "content": self._user_prompt(text, title, timing_hint)},
        ]
        return messages, {"words": words, "n_pause": n_pause}

    def detect(self, transcript: dict, title: Optional[str] = None) -> dict:
        messages, aux = self.prepare(transcript, title)
        if messages is None:
            return aux
        return self.finish(self._post(messages), aux)

    def finish(self, out: Optional[dict], aux: dict) -> dict:
        """Post-LLM half: turn a parsed LLM JSON dict into the detect() result
        (quote location, scene block, narration override). out=None -> error."""
        words, n_pause = aux["words"], aux["n_pause"]
        if out is None:
            return {"is_target": "error", "hard_negative_type": None, "matches": []}

        is_target = out.get("is_target", "no")
        hard_neg = out.get("hard_negative_type", "none")
        if is_target != "yes" or hard_neg not in (None, "none"):
            return {"is_target": is_target, "hard_negative_type": hard_neg,
                    "modality": None, "agent": None, "matches": []}
        modality = out.get("modality") or "witnessed"
        agent = out.get("agent") or "human"
        scene = {
            "scene_type": out.get("scene_type"),       # action | narration | mixed
            "n_people": out.get("n_people"),
            "violator_role": out.get("violator_role"),  # subject|camera_person|third_party|contested
            "reactor_role": out.get("reactor_role"),    # bystander|camera_person|victim|mixed
            "severity": out.get("severity"),             # 1-5 flagrancy of the violation
            "reaction_strength": out.get("reaction_strength"),  # 1-5 viscerality of the reaction
            "n_speech_pauses": n_pause,                  # computed (turn-taking proxy)
        }
        # HARD RULE (measured ~20% witnessed FP, mostly solo narration): a lone
        # narrator describing a violation is commentary no matter what STEP 2 said.
        try:
            n_people = int(scene.get("n_people") or 0)
        except (TypeError, ValueError):
            n_people = 0
        if (modality == "witnessed" and scene.get("scene_type") == "narration"
                and n_people <= 1):
            log.info("narration+solo override: witnessed -> commentary")
            modality = "commentary"

        wt = self._word_tokens(words)
        matches = []
        for rx in out.get("reactions", []):
            quote = (rx.get("quote") or "").strip()
            loc = self._locate(quote, wt)
            if not loc:
                continue  # unlocatable -> can't clip
            start, end, speaker = loc
            norm = rx.get("norm")
            matches.append({
                "phrase": quote[:120],
                "matched_text": quote,
                "tier": LLM_TIER,
                "tag": rx.get("signal"),
                "norm": norm,
                "start": start,
                "end": end,
                "speaker": speaker,
                "context": (f"[{norm}] " if norm else "") + self._context(wt, start, end),
            })
        expand = [q.strip() for q in (out.get("expand_queries") or [])
                  if isinstance(q, str) and 2 <= len(q.strip()) <= 80][:3]
        return {"is_target": "yes", "hard_negative_type": "none",
                "modality": modality, "agent": agent, "scene": scene,
                "expand_queries": expand, "matches": matches}


# --------------------------------------------------------------------------- #
# Module-level quote->timestamp helpers (shared by the instructional detector).
# --------------------------------------------------------------------------- #
def _wt_tokens(words: list) -> list:
    toks = []
    for w in words:
        t = _norm_token(w.get("word", ""))
        if not t or w.get("start") is None:
            continue
        toks.append((t, float(w["start"]), float(w["end"]), w.get("speaker")))
    return toks


def _locate_quote(quote: str, wt: list):
    """Return (start,end) of a quote's token span in the word stream, or None."""
    q = [_norm_token(x) for x in (quote or "").split()]
    q = [x for x in q if x]
    if len(q) < 2:
        return None
    toks = [t[0] for t in wt]
    for L in range(len(q), 1, -1):
        sub = q[:L]
        for i in range(0, len(toks) - L + 1):
            if toks[i:i + L] == sub:
                return wt[i][1], wt[i + L - 1][2]
    return None


# =========================================================================== #
# Instructional / demonstrated-norm detector (separate corpus).
# The candid detector reads a bystander REACTION as a post-hoc, implicit signal;
# here an educator EXPLICITLY explains the norm and a role-play DEMO enacts it,
# so the explanation IS the ground-truth label. See docs/instructional_norm_vein.md
# =========================================================================== #
SYS_INSTR = """You analyze a video transcript (ASR, may be messy; speaker-tagged if available) plus its title to decide whether it is an INSTRUCTIONAL social-norm / social-skills video -- a teacher/narrator explicitly TEACHES a social norm and a role-play DEMO enacts it (autism/SEL social-skills lessons, "expected vs unexpected behavior", social stories, corporate harassment/DEI/bystander training).
A DEMO/role-play = enacted in-character dialogue (names, situated/quoted speech) showing the norm violated or followed. EXPLANATION = the educator addressing the audience, naming/judging the behavior ("that was unexpected because...", "this is an example of quid pro quo harassment", "a better choice would be..."). Framing markers ("let's watch", "in this scenario", "now let's discuss", "did you see that?") bracket demos.
If NOT instructional (candid/unscripted, lecture with no demo, unrelated), set is_instructional="no".
Quote VERBATIM from the transcript so quotes can be located; use null if absent; do NOT invent. Output ONLY JSON."""


class InstructionalDetector:
    """Detect didactic norm videos + extract (norm, demo span, explanation) tuples."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        lc = cfg.get("llm", {})
        self.endpoint = lc.get("endpoint", "http://127.0.0.1:8001/v1/chat/completions")
        self.model = lc.get("model", "llama70b")
        self.timeout = lc.get("timeout", 180)
        self.max_tokens = lc.get("max_tokens", 1600)
        self.max_chars = lc.get("max_chars", 8000)

    def _user_prompt(self, text: str, title: Optional[str]) -> str:
        return (
            (f'Video title: "{title}"\n\n' if title else "")
            + f'Transcript:\n"""{text[:self.max_chars]}"""\n\n'
            'Return JSON:\n'
            '{"is_instructional":"yes"|"no",'
            '"genre":"autism_sel"|"social_story"|"harassment_prevention"|"dei_bias"|"bystander"|"workplace"|"other",'
            '"norms":[{"statement":"<canonical norm, e.g. Ask before borrowing>","category":"<short>"}],'
            '"demos":[{"polarity":"violation"|"correct"|"contrast","norm":"<short>",'
            '"start_quote":"<exact first words of the enacted scene>",'
            '"end_quote":"<exact last words of the enacted scene>",'
            '"explanation":"<exact verbatim quote where the educator names/judges what happened -- the LABEL>"}],'
            '"expected_behavior":"<short statement of the recommended behavior, or null>"}\n'
            'List up to 12 demos. Verbatim quotes only. JSON only.')

    def _post(self, messages: list[dict]) -> Optional[dict]:
        try:
            r = requests.post(self.endpoint, json={
                "model": self.model,
                "messages": messages,
                "temperature": 0, "max_tokens": self.max_tokens,
                "response_format": {"type": "json_object"},
            }, timeout=self.timeout)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            return LLMDetector._parse(content)
        except Exception as e:
            log.warning("instructional LLM call failed: %s", str(e)[:160])
            return None

    def prepare(self, transcript: dict, title: Optional[str] = None):
        """(messages, aux) for the batch engine; (None, result) if no call needed."""
        words = transcript.get("words", [])
        text = " ".join(w.get("word", "").strip() for w in words).strip()
        if len(text.split()) < 10:
            return None, {"is_instructional": "no"}
        messages = [{"role": "system", "content": SYS_INSTR},
                    {"role": "user", "content": self._user_prompt(text, title)}]
        return messages, {"words": words}

    def detect(self, transcript: dict, title: Optional[str] = None) -> dict:
        messages, aux = self.prepare(transcript, title)
        if messages is None:
            return aux
        return self.finish(self._post(messages), aux)

    def finish(self, out: Optional[dict], aux: dict) -> dict:
        words = aux["words"]
        if out is None:
            return {"is_instructional": "error"}
        if out.get("is_instructional") != "yes":
            return {"is_instructional": out.get("is_instructional", "no")}
        wt = _wt_tokens(words)
        demos = []
        for d in out.get("demos", []):
            loc = _locate_quote(d.get("start_quote", ""), wt)
            end = _locate_quote(d.get("end_quote", ""), wt)
            demos.append({
                "polarity": d.get("polarity"),
                "norm": d.get("norm"),
                "start_quote": d.get("start_quote"),
                "end_quote": d.get("end_quote"),
                "explanation": d.get("explanation"),
                "start": loc[0] if loc else None,
                "end": (end[1] if end else (loc[1] if loc else None)),
            })
        return {
            "is_instructional": "yes",
            "genre": out.get("genre"),
            "norms": out.get("norms", []),
            "demos": demos,
            "expected_behavior": out.get("expected_behavior"),
        }
