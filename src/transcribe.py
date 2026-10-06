"""WhisperX transcription with word-level alignment (+ optional diarization).

Heavy deps (torch / whisperx) are imported lazily inside the class so the rest
of the pipeline (detection, state, query gen) stays importable on machines
without a GPU build installed.

Output JSON (saved for EVERY video, hit or miss) has the shape:
    {
      "video_id": str,
      "language": str,
      "segments": [{start, end, text, speaker?, words: [{word,start,end,score,speaker?}]}],
      "words": [ flattened word list with timings ]   # convenience for the matcher
    }
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from . import state

log = logging.getLogger("transcribe")


def _pseudo_words(seg: dict) -> list[dict]:
    """Approximate per-word timings for a segment WhisperX could not align
    (a language with no wav2vec2 alignment model). Spreads [seg.start, seg.end]
    linearly over the segment's tokens so the transcript text is preserved and
    quotes still locate at coarse (segment) granularity. Whitespace-tokenized;
    for space-less scripts (CJK/Thai) we fall back to per-character tokens so a
    quote of >=2 units can still be matched."""
    text = (seg.get("text") or "").strip()
    start, end = seg.get("start"), seg.get("end")
    if not text or start is None or end is None:
        return []
    start, end = float(start), float(end)
    toks = text.split()
    if len(toks) <= 1 and len(text) > 1:
        toks = list(text)            # space-less script -> per-character
    n = len(toks)
    if n == 0:
        return []
    step = max(end - start, 0.0) / n
    out = []
    for i, tok in enumerate(toks):
        ws = start + i * step
        we = end if i == n - 1 else start + (i + 1) * step
        out.append({"word": tok, "start": round(ws, 3),
                    "end": round(we, 3), "score": None, "speaker": None})
    return out


class Transcriber:
    """Loads the WhisperX model(s) once and reuses them across videos."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.tc = cfg["transcribe"]
        self._model = None
        self._align = None        # (align_model, metadata)
        self._diarize = None
        self._align_lang = None
        self._device_index = self._resolve_device_index()
        # device string for align/diarize/load_audio (load_model takes them split)
        self._device = (f"{self.tc['device']}:{self._device_index}"
                        if self.tc.get("device") == "cuda" else self.tc.get("device", "cpu"))

    def _resolve_device_index(self) -> int:
        """Honor an explicit device_index, else auto-pick the freest GPU.

        sk3 is shared; the 'idle' GPU moves around, so for device_index in
        (null, 'auto', -1) we query nvidia-smi and choose the one with the most
        free memory at startup.
        """
        di = self.tc.get("device_index", "auto")
        if isinstance(di, int) and di >= 0:
            return di
        if self.tc.get("device") != "cuda":
            return 0
        import subprocess
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=30).stdout
            best_idx, best_free = 0, -1
            for line in out.strip().splitlines():
                idx, free = (x.strip() for x in line.split(","))
                if int(free) > best_free:
                    best_idx, best_free = int(idx), int(free)
            log.info("auto-selected GPU %d (%d MiB free)", best_idx, best_free)
            return best_idx
        except Exception as e:
            log.warning("GPU auto-select failed (%s); using device 0", e)
            return 0

    # -- lazy loaders --------------------------------------------------------
    def _load_asr(self):
        if self._model is None:
            import whisperx  # noqa: WPS433 (lazy)
            log.info("loading WhisperX %s on %s:%d (%s)", self.tc["model"],
                     self.tc["device"], self._device_index, self.tc["compute_type"])
            self._model = whisperx.load_model(
                self.tc["model"],
                device=self.tc["device"],
                device_index=self._device_index,
                compute_type=self.tc["compute_type"],
                language=self.tc.get("language"),
            )
        return self._model

    def _load_align(self, language_code: str):
        # Cache one alignment model at a time (bounded GPU memory on shared sk3).
        # None is a VALID cached value meaning "no wav2vec2 model for this
        # language" -> fall back to segment-level timings rather than crash the
        # batch. Keyed on language change only (so a cached None isn't retried).
        if self._align_lang != language_code:
            import whisperx
            try:
                self._align = whisperx.load_align_model(
                    language_code=language_code, device=self._device)
            except Exception as e:
                log.warning("no alignment model for language %r (%s); "
                            "segment-level timestamps only", language_code, e)
                self._align = None
            self._align_lang = language_code
        return self._align

    def _load_diarize(self):
        if self._diarize is None:
            import whisperx
            token = os.environ.get(self.tc.get("hf_token_env", "HF_TOKEN"))
            if not token:
                log.warning("diarization requested but HF token env unset; skipping")
                return None
            self._diarize = whisperx.DiarizationPipeline(
                use_auth_token=token, device=self._device)
        return self._diarize

    # -- main entry ----------------------------------------------------------
    def transcribe(self, media_path: Path, video_id: str) -> dict:
        import whisperx

        device = self._device
        audio = whisperx.load_audio(str(media_path))

        model = self._load_asr()
        result = model.transcribe(audio, batch_size=self.tc.get("batch_size", 16))
        language = result.get("language", self.tc.get("language") or "en")

        if self.tc.get("align", True):
            aligned = self._load_align(language)
            if aligned is not None:
                model_a, meta = aligned
                result = whisperx.align(
                    result["segments"], model_a, meta, audio, device,
                    return_char_alignments=False)
            else:
                log.info("%s: language %s has no alignment model; "
                         "using segment-level timings", video_id, language)

        if self.tc.get("diarize", False):
            dia = self._load_diarize()
            if dia is not None:
                try:
                    diarize_df = dia(audio)
                    result = whisperx.assign_word_speakers(diarize_df, result)
                except Exception as e:  # diarization is best-effort
                    log.warning("diarization failed for %s: %s", video_id, e)

        out = self._normalize(result, video_id, language)
        self.save(out, video_id)
        return out

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _normalize(result: dict, video_id: str, language: str) -> dict:
        segments = result.get("segments", [])
        words: list[dict] = []
        for seg in segments:
            timed = [w for w in (seg.get("words") or [])
                     if w.get("start") is not None and w.get("end") is not None]
            if timed:
                for w in timed:
                    words.append({
                        "word": w.get("word", "").strip(),
                        "start": float(w["start"]),
                        "end": float(w["end"]),
                        "score": w.get("score"),
                        "speaker": w.get("speaker"),
                    })
            else:
                # segment WhisperX could not align (language without a model):
                # synthesize coarse timings so the text isn't lost downstream.
                words.extend(_pseudo_words(seg))
        return {
            "video_id": video_id,
            "language": language,
            "segments": segments,
            "words": words,
        }

    def save(self, transcript: dict, video_id: str) -> Path:
        out_dir = state.resolve_path(self.cfg["paths"]["transcripts"])
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{video_id}.json"
        out.write_text(json.dumps(transcript, ensure_ascii=False))
        return out

    def load_existing(self, video_id: str) -> Optional[dict]:
        out = state.resolve_path(self.cfg["paths"]["transcripts"]) / f"{video_id}.json"
        if out.exists():
            return json.loads(out.read_text())
        return None
