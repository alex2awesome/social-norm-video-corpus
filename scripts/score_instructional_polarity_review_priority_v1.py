#!/usr/bin/env python3
"""Score audited instructional polarity review priority over an existing corpus.

Every demo is emitted.  Non-explanation polarity receives earlier manual-review
priority; explanation and unknown polarity remain preserved and eligible for
the independent visual-demo contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PRIORITY_POLARITIES = {"violation", "correct", "contrast"}


def score_demo(uid: str, index: int, demo: dict[str, Any]) -> dict[str, Any]:
    polarity = str(demo.get("polarity") or "unknown").strip().casefold()
    prioritized = polarity in PRIORITY_POLARITIES
    return {
        "uid": uid,
        "demo_index": index,
        "clip": demo.get("clip"),
        "polarity": polarity,
        "instructional_non_explanation_review_priority_v1": polarity if prioritized else None,
        "priority_tier": "standard_above_explanation" if prioritized else "lower_priority_preserved",
        "review_route": "instructional_demo_manual_review",
        "automatic_acceptance": False,
        "automatic_rejection": False,
        "delete_media": False,
        "policy": "review_order_only_preserve_every_demo",
        "error": None,
    }


def score_corpus(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted((root / "data" / "instructional").glob("*/metadata.json")):
        uid = metadata_path.parent.name
        try:
            payload = json.loads(metadata_path.read_text())
            demos = payload.get("demos") or []
            if not isinstance(demos, list):
                raise ValueError("demos is not a list")
            rows.extend(score_demo(uid, index, demo) for index, demo in enumerate(demos))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            rows.append({
                "uid": uid,
                "demo_index": None,
                "clip": None,
                "polarity": None,
                "instructional_non_explanation_review_priority_v1": None,
                "priority_tier": "unscored_preserved",
                "review_route": "instructional_demo_manual_review",
                "automatic_acceptance": False,
                "automatic_rejection": False,
                "delete_media": False,
                "policy": "review_order_only_preserve_every_demo",
                "error": f"{type(error).__name__}: {error}",
            })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"output exists: {args.out}")
    rows = score_corpus(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    successful = [row for row in rows if row["error"] is None]
    summary = {
        "rows": len(rows),
        "successful_demo_rows": len(successful),
        "failed_source_rows": len(rows) - len(successful),
        "prioritized_demo_rows": sum(
            row["instructional_non_explanation_review_priority_v1"] is not None
            for row in successful
        ),
        "lower_priority_preserved_demo_rows": sum(
            row["instructional_non_explanation_review_priority_v1"] is None
            for row in successful
        ),
        "automatic_acceptance": False,
        "automatic_rejection": False,
        "corpus_mutated": False,
        "out": str(args.out),
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
