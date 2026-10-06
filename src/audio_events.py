"""Audio-event channel: locate screams / gasps / commotion in video audio with
PANNs CNN14 sound-event detection (AudioSet, 527 classes).

Why: the transcript pipeline only sees VERBAL reactions; the most visceral
bystander reactions are often non-verbal (a scream, a gasp, a crowd erupting,
glass shattering). This channel scores those directly from the waveform. It
serves two roles:
  1. interestingness reranker for existing hits (Pass A, batch_audio_events)
  2. an independent recall channel for reaction moments the keyword/LLM
     pipeline missed (Pass B, wired into the batch pipeline later)

Model: Cnn14_DecisionLevelMax via `panns_inference` -- framewise (~10ms)
sigmoid scores per AudioSet class; checkpoint (~440MB) auto-downloads to
$HOME/panns_data on first use. ~2GB VRAM on GPU, also runs on CPU (slow).

Scores are AudioSet-weak-label quality (per-class AP ~0.4-0.6 for the scream/
cry classes): treat them as a RANKING signal, not ground truth.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger("audio_events")

# AudioSet class names (exact panns labels, matched case-insensitively) we
# track, grouped by what they mean for us. vocal_reaction = a person/crowd
# audibly reacting; commotion = the violent/chaotic event sounds themselves.
GROUPS = {
    "vocal_reaction": [
        "Screaming", "Shout", "Yell", "Bellow", "Children shouting",
        "Crying, sobbing", "Wail, moan", "Groan", "Gasp", "Whoop",
        "Cheering",
    ],
    "commotion": [
        "Crowd", "Hubbub, speech noise, speech babble", "Slap, smack",
        "Whack, thwack", "Smash, crash", "Breaking", "Shatter", "Glass",
        "Gunshot, gunfire", "Skidding", "Tire squeal", "Squeal",
        "Car alarm", "Siren",
    ],
}


class AudioEventDetector:
    SR = 32000   # panns models are trained at 32kHz

    def __init__(self, cfg: dict):
        acfg = cfg.get("audio_events", {})
        self.threshold = float(acfg.get("threshold", 0.30))
        self.chunk_sec = int(acfg.get("chunk_sec", 60))
        self.overlap_sec = int(acfg.get("overlap_sec", 5))
        self.ffmpeg = shutil.which("ffmpeg") or "ffmpeg"

        import torch
        from panns_inference import SoundEventDetection
        from panns_inference.config import labels

        want_cuda = acfg.get("device", "cuda") == "cuda"
        self.device = "cuda" if (want_cuda and torch.cuda.is_available()) else "cpu"
        self.sed = SoundEventDetection(checkpoint_path=None, device=self.device)

        # map our class names -> (group, AudioSet column index)
        low = {l.lower(): i for i, l in enumerate(labels)}
        self.idx: dict[str, tuple[str, int]] = {}
        for grp, names in GROUPS.items():
            for n in names:
                i = low.get(n.lower())
                if i is None:
                    log.warning("AudioSet label not found, skipping: %r", n)
                else:
                    self.idx[n] = (grp, i)
        log.info("audio-event detector up on %s: %d/%d target classes resolved",
                 self.device, len(self.idx), sum(len(v) for v in GROUPS.values()))

    # ------------------------------------------------------------------ audio
    def _decode(self, path: Path) -> np.ndarray:
        """Decode any container to 32kHz mono float32 via ffmpeg."""
        cmd = [self.ffmpeg, "-v", "error", "-i", str(path), "-vn",
               "-ac", "1", "-ar", str(self.SR), "-f", "f32le", "pipe:1"]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
        return np.frombuffer(out, dtype=np.float32).copy()

    # --------------------------------------------------------------- analysis
    def analyze(self, path: Path) -> dict | None:
        """Full-file SED -> per-second class scores -> thresholded events.

        Returns {"duration", "events", "class_peaks", "summary"} or None for
        audio shorter than 1s / undecodable.
        """
        try:
            audio = self._decode(path)
        except subprocess.CalledProcessError as e:
            log.warning("ffmpeg failed on %s: %s", path.name,
                        e.stderr.decode(errors="replace")[:200])
            return None
        dur = len(audio) / self.SR
        if dur < 1.0:
            return None
        n_sec = int(np.ceil(dur))
        per_sec = {name: np.zeros(n_sec, np.float32) for name in self.idx}

        # chunked SED (overlap so events on a boundary aren't halved); per-
        # second score = max framewise sigmoid inside that second
        step = (self.chunk_sec - self.overlap_sec) * self.SR
        for start in range(0, len(audio), step):
            seg = audio[start: start + self.chunk_sec * self.SR]
            if len(seg) < self.SR // 2 and start > 0:
                break
            fw = self.sed.inference(seg[None, :])[0]          # (frames, 527)
            fps = fw.shape[0] / (len(seg) / self.SR)
            sec0 = start // self.SR
            for name, (_grp, ci) in self.idx.items():
                col = fw[:, ci]
                for s in range(int(np.ceil(len(seg) / self.SR))):
                    a, b = int(s * fps), max(int((s + 1) * fps), int(s * fps) + 1)
                    if a >= len(col) or sec0 + s >= n_sec:
                        break
                    v = float(col[a:b].max())
                    if v > per_sec[name][sec0 + s]:
                        per_sec[name][sec0 + s] = v

        events = []
        for name, (grp, _ci) in self.idx.items():
            for t0, t1 in self._runs(per_sec[name], self.threshold):
                events.append({
                    "class": name, "group": grp, "t0": t0, "t1": t1,
                    "peak": round(float(per_sec[name][t0:t1].max()), 4),
                })
        events.sort(key=lambda e: (e["t0"], -e["peak"]))

        class_peaks = {n: round(float(a.max()), 4) for n, a in per_sec.items()
                       if a.max() >= 0.05}
        vocal_peak = max([a.max() for n, a in per_sec.items()
                          if self.idx[n][0] == "vocal_reaction"], default=0.0)
        comm_peak = max([a.max() for n, a in per_sec.items()
                         if self.idx[n][0] == "commotion"], default=0.0)
        summary = {
            "scream_peak": round(float(vocal_peak), 4),
            "commotion_peak": round(float(comm_peak), 4),
            # single rank key: vocal reactions are the prize; commotion alone
            # (sirens, crowds) is supporting evidence, so it is discounted
            "interest": round(float(max(vocal_peak, 0.7 * comm_peak)), 4),
            "n_events": len(events),
        }
        return {"duration": round(dur, 2), "events": events,
                "class_peaks": class_peaks, "summary": summary}

    @staticmethod
    def _runs(arr: np.ndarray, thr: float, max_gap: int = 1) -> list[tuple[int, int]]:
        """Contiguous >=thr second-runs, merging gaps <= max_gap seconds."""
        runs, s = [], None
        for i, v in enumerate(arr):
            if v >= thr and s is None:
                s = i
            elif v < thr and s is not None:
                runs.append((s, i))
                s = None
        if s is not None:
            runs.append((s, len(arr)))
        merged: list[tuple[int, int]] = []
        for r in runs:
            if merged and r[0] - merged[-1][1] <= max_gap:
                merged[-1] = (merged[-1][0], r[1])
            else:
                merged.append(r)
        return merged
