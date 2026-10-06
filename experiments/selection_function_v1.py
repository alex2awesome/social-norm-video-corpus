#!/usr/bin/env python3
"""S1: first-pass observational estimates of the reaction selection function.

Cross-tabulates detector-era reaction strength against severity, head count,
norm domain, content type, and platform over the witnessed set, and compares
witnessed vs commentary norm-domain composition (the reaction /
no-in-situ-reaction partition).  Pure ledger analysis — no GPU, no media.
All inputs are weak labels; treat outputs as ranking-quality evidence until
gold calibration exists.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

ANALYSIS_VERSION = "selection_function_v1"


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def analyze(root: Path, source_context: Path, proposals: Path) -> dict[str, Any]:
    context = {}
    commentary_domains: Counter = Counter()
    for row in iter_jsonl(source_context):
        result = row.get("result")
        if not result:
            continue
        context[row["uid"]] = result
        if "commentary" in (row.get("pillars") or []):
            commentary_domains[result["norm_domain"]] += 1

    strength_by: dict[str, dict[Any, list[float]]] = defaultdict(lambda: defaultdict(list))
    witnessed_domains: Counter = Counter()
    seen = set()
    for row in iter_jsonl(proposals):
        uid = row["uid"]
        if uid in seen:
            continue
        seen.add(uid)
        try:
            metadata = json.loads((root / "data" / "hits" / uid / "metadata.json").read_text())
        except (OSError, json.JSONDecodeError):
            continue
        scene = (metadata.get("provenance") or {}).get("scene") or {}
        strength = scene.get("reaction_strength")
        label = context.get(uid) or {}
        if label.get("norm_domain"):
            witnessed_domains[label["norm_domain"]] += 1
        if not isinstance(strength, (int, float)):
            continue
        strength = float(strength)
        if isinstance(scene.get("severity"), (int, float)):
            strength_by["severity"][int(scene["severity"])].append(strength)
        n_people = scene.get("n_people")
        if isinstance(n_people, (int, float)):
            bucket = "1" if n_people <= 1 else "2" if n_people == 2 else "3-5" if n_people <= 5 else "6+"
            strength_by["n_people_bucket"][bucket].append(strength)
        if label.get("norm_domain"):
            strength_by["norm_domain"][label["norm_domain"]].append(strength)
        if label.get("content_type"):
            strength_by["content_type"][label["content_type"]].append(strength)
        strength_by["platform"][uid.split("__")[0]].append(strength)

    report: dict[str, Any] = {"analysis_version": ANALYSIS_VERSION,
                              "witnessed_sources": len(seen)}
    for axis, groups in strength_by.items():
        report[f"mean_reaction_strength_by_{axis}"] = {
            str(key): {"mean": mean(values), "n": len(values)}
            for key, values in sorted(groups.items(), key=lambda kv: str(kv[0]))
        }
    total_w = sum(witnessed_domains.values()) or 1
    total_c = sum(commentary_domains.values()) or 1
    report["norm_domain_partition"] = {
        domain: {
            "witnessed_share": round(witnessed_domains.get(domain, 0) / total_w, 3),
            "commentary_share": round(commentary_domains.get(domain, 0) / total_c, 3),
        }
        for domain in sorted(set(witnessed_domains) | set(commentary_domains))
    }
    report["caveats"] = [
        "severity/strength are detector-assigned (uncalibrated, top-heavy)",
        "platform 85% dailymotion; query-lineage selection uncontrolled",
        "commentary composition reflects retrieval, not population",
    ]
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-context", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"output exists: {args.out}")
    report = analyze(args.root, args.source_context, args.proposals)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in report.items()
                      if k in ("witnessed_sources", "mean_reaction_strength_by_severity")},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
