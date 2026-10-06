#!/usr/bin/env python3
"""Append a separately audited V9A record for one missing terminal JSON brace.

This does not rerun a model or reinterpret its content. It accepts only a raw
response with an opening object, no closing object, and a schema-valid V9A JSON
object after appending exactly one ``}``. The original failed attempts remain
unchanged in the append-only input/output ledger.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from scripts.run_open_vlm_scene_benchmark import parse_json
else:
    from run_open_vlm_scene_benchmark import parse_json


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def repair(record: dict) -> dict:
    if record.get("result") is not None or not record.get("error"):
        raise ValueError("record is not a failed V9A attempt")
    if record.get("rubric") != "v9a_storyboard":
        raise ValueError("record is not a V9A storyboard attempt")
    raw = str(record.get("raw_response") or "").strip()
    if not raw.startswith("{") or "}" in raw:
        raise ValueError("repair requires one unterminated top-level object")
    parsed = parse_json(raw + "}", "v9a")
    parsed["syntax_validation_repair"] = (
        "appended_exactly_one_missing_terminal_object_brace"
    )
    return {
        **record,
        "result": parsed,
        "error": None,
        "repair_provenance": {
            "kind": "single_terminal_brace_syntax_repair",
            "model_rerun": False,
            "semantic_fields_changed": False,
            "raw_response_changed": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--item-id", required=True)
    args = parser.parse_args()
    rows = [
        row
        for row in load_jsonl(args.ledger)
        if row.get("item_id") == args.item_id
    ]
    if not rows:
        raise SystemExit("item_id not found")
    repaired = repair(rows[-1])
    with args.ledger.open("a") as handle:
        handle.write(json.dumps(repaired, sort_keys=True) + "\n")
    print(json.dumps({"item_id": args.item_id, "repaired": True}, sort_keys=True))


if __name__ == "__main__":
    main()
