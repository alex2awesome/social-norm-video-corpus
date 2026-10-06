import json, os, random
random.seed(13)
DISC = "data/discussion"; TRANS = "data/transcripts"
by_cat = {}
for f in os.listdir(DISC):
    if not f.endswith(".json"): continue
    try: d = json.load(open(os.path.join(DISC, f)))
    except: continue
    cat = d.get("category") or "None"
    if cat.startswith("comm_"):
        by_cat.setdefault(cat, []).append((f[:-5], d))
print("comm_* commentary extractions in data/discussion/:", {c: len(v) for c, v in sorted(by_cat.items())} or "NONE YET")
def ctx(uid, q, w=100):
    if not q: return "(no quote)"
    tp = os.path.join(TRANS, uid+".json")
    if not os.path.exists(tp): return "(no transcript)"
    try: tr=json.load(open(tp))
    except: return "(bad)"
    seg = tr.get("segments",[]) if isinstance(tr,dict) else tr
    t = " ".join(s.get("text","") for s in seg) if isinstance(seg,list) else str(tr)
    i = t.lower().find(" ".join(q.split()[:6]).lower())
    if i<0: return "*** NOT FOUND ***"
    return t[max(0,i-w):i+len(q)+w].replace("\n"," ")
for cat, items in sorted(by_cat.items()):
    random.shuffle(items)
    print("\n=== %s (showing 3) ===" % cat)
    for uid, d in items[:3]:
        print("  ### %s | %s" % (uid, (d.get("title") or "?")[:70]))
        for s in (d.get("statements") or [])[:2]:
            print("     [%s/%s] %r" % (s.get("signal"), s.get("norm"), (s.get("quote") or "")[:55]))
            print("        tctxt:", ctx(uid, s.get("quote"))[:160])
