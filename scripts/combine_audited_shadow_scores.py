#!/usr/bin/env python3
"""Apply the audited registry to an append-only JSONL shadow score stream.

Only fields whose names exactly equal registered rule IDs are considered.
Failed-transfer rules abstain through ``combine_signals``.  Input media and
metadata are never changed; each input row produces one output row with the
audited review routes, rankings, and strict-route exclusions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

try:
    # Import used by tests and ``python -m scripts...``.
    from scripts.weak_signal_registry import combine_signals, load_registry, validate_registry
except ModuleNotFoundError:  # pragma: no cover - exercised by direct CLI smoke tests
    # Direct ``python scripts/combine_audited_shadow_scores.py`` execution puts
    # the scripts directory, not the repository root, on sys.path.
    from weak_signal_registry import combine_signals, load_registry, validate_registry


IDENTIFIER_FIELDS = ("uid", "demo_index", "clip", "source_uid", "item_id")


def registered_outputs(
    row: dict[str, Any], registry: dict[str, Any], pillar: str
) -> dict[str, Any]:
    rule_ids = {
        rule["rule_id"]
        for rule in registry["rules"]
        if rule.get("pillar") in {pillar, "all"}
    }
    return {key: value for key, value in row.items() if key in rule_ids}


def combine_row(
    row: dict[str, Any], registry: dict[str, Any], pillar: str
) -> dict[str, Any]:
    if pillar not in {"instructional", "commentary", "witnessed", "all"}:
        raise ValueError(f"invalid pillar: {pillar!r}")
    outputs = registered_outputs(row, registry, pillar)
    return {
        **{key: row.get(key) for key in IDENTIFIER_FIELDS if key in row},
        "pillar": pillar,
        "signal_outputs": outputs,
        "audited_combination": combine_signals(outputs, registry),
        "source_error": row.get("error"),
        "input_preserved": True,
    }


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            yield line_number, value


def combine_file(
    input_path: Path,
    output_path: Path,
    registry: dict[str, Any],
    pillar: str,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(f"output exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    triggered_rows = 0
    excluded_rows = 0
    routed_rows = 0
    ranking_rows = 0
    with output_path.open("x") as output:
        for _, row in iter_jsonl(input_path):
            combined = combine_row(row, registry, pillar)
            decision = combined["audited_combination"]
            rows += 1
            triggered_rows += bool(decision["triggered_rule_ids"])
            excluded_rows += bool(decision["strict_route_exclusions"])
            routed_rows += bool(decision["review_routes"])
            ranking_rows += bool(decision["rankings"])
            output.write(json.dumps(combined, sort_keys=True) + "\n")
    return {
        "rows": rows,
        "pillar": pillar,
        "triggered_rows": triggered_rows,
        "routed_rows": routed_rows,
        "ranking_rows": ranking_rows,
        "strict_route_exclusion_rows": excluded_rows,
        "acceptance_rows": 0,
        "rejection_rows": 0,
        "deleted_rows": 0,
        "input_mutated": False,
        "output": str(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument(
        "--pillar",
        choices=("instructional", "commentary", "witnessed", "all"),
        required=True,
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    registry = load_registry(args.registry)
    validate_registry(registry, args.root)
    summary = combine_file(args.input, args.out, registry, args.pillar)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
