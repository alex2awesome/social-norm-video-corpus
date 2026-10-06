"""Positive-control check for the audio-event channel: run PANNs SED on a few
GOLD hits (severity>=4, reaction_strength=5 -- videos the LLM says have a
flagrant violation + visceral reaction) and print what fires. If the channel
is any good, these should light up where the quiet smoke-test videos didn't.

    CUDA_VISIBLE_DEVICES=<gpu> ./run.sh python -u scripts/validate_audio_events.py
"""
import json
import sys

sys.path.insert(0, ".")
from src import state                                    # noqa: E402
from src.batch_transcribe import _find_raw               # noqa: E402

cfg = state.load_config()
hits_dir = state.resolve_path(cfg["paths"]["hits"])

gold = []
for d in sorted(hits_dir.iterdir()):
    mp = d / "metadata.json"
    if not mp.exists():
        continue
    try:
        m = json.loads(mp.read_text())
    except json.JSONDecodeError:
        continue
    sc = (m.get("provenance") or {}).get("scene") or {}
    if (sc.get("severity") or 0) >= 4 and (sc.get("reaction_strength") or 0) >= 5:
        raw = _find_raw(cfg, d.name)
        if raw is not None:
            gold.append((d.name, m.get("title", "")[:70], raw))
    if len(gold) >= 4:
        break

print(f"positive controls: {len(gold)} gold hits with raws")
cfg.setdefault("audio_events", {})["threshold"] = 0.12   # calibration trial
from src.audio_events import AudioEventDetector          # noqa: E402
det = AudioEventDetector(cfg)
for uid, title, raw in gold:
    r = det.analyze(raw)
    if r is None:
        print(f"\n{uid}  DECODE FAILED")
        continue
    s = r["summary"]
    meta = json.loads((hits_dir / uid / "metadata.json").read_text())
    react_ts = sorted(round(x.get("start", -1)) for x in meta.get("reactions", []))
    print(f"\n{uid}  ({r['duration']:.0f}s)  {title!r}")
    print(f"  LLM reaction timestamps: {react_ts}")
    print(f"  scream_peak={s['scream_peak']}  commotion_peak={s['commotion_peak']}"
          f"  interest={s['interest']}  events={s['n_events']}")
    for e in r["events"][:10]:
        near = any(e["t0"] - 8 <= t <= e["t1"] + 8 for t in react_ts)
        print(f"    {e['t0']:>5}-{e['t1']:<5} {e['class']:<28} peak={e['peak']}"
              f"{'   <-- near LLM reaction' if near else ''}")
