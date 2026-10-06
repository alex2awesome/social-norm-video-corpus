#!/usr/bin/env python3
"""Audit ECAPA speaker-change proxies on the frozen witnessed holdout."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

if __package__:
    from scripts.evaluate_witnessed_reaction_external_holdout import manual_target
    from scripts.evaluate_witnessed_reaction_candidate_vlm import candidate_positive
else:
    from evaluate_witnessed_reaction_external_holdout import manual_target
    from evaluate_witnessed_reaction_candidate_vlm import candidate_positive


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_candidate_ledger(
    voice_rows: list[dict[str, Any]], ledger: list[dict[str, str]]
) -> dict[str, Any]:
    expected = {str(row["candidate_id"]) for row in voice_rows}
    observed = {row["candidate_id"] for row in ledger}
    if len(observed) != len(ledger):
        raise ValueError("duplicate candidate_id in manual ledger")
    if observed != expected:
        raise ValueError(
            f"manual candidate coverage mismatch: missing={sorted(expected-observed)}, "
            f"extra={sorted(observed-expected)}"
        )
    return {
        "manual_candidate_rows": len(ledger),
        "manual_candidate_coverage_complete": True,
        "manual_visual_identity_roles": dict(
            Counter(row["visual_identity_role"] for row in ledger)
        ),
        "manual_proxy_dispositions": dict(
            Counter(row["speaker_proxy_disposition"] for row in ledger)
        ),
    }


def metrics(gold: dict[str, bool], predicted: dict[str, bool]) -> dict[str, Any]:
    item_ids = sorted(gold)
    tp = [item for item in item_ids if gold[item] and predicted.get(item, False)]
    fp = [item for item in item_ids if not gold[item] and predicted.get(item, False)]
    fn = [item for item in item_ids if gold[item] and not predicted.get(item, False)]
    tn = [item for item in item_ids if not gold[item] and not predicted.get(item, False)]
    return {
        "items": len(item_ids),
        "positives": sum(gold.values()),
        "selected": len(tp) + len(fp),
        "tp": len(tp),
        "fp": len(fp),
        "fn": len(fn),
        "tn": len(tn),
        "precision": len(tp) / (len(tp) + len(fp)) if tp or fp else None,
        "recall": len(tp) / (len(tp) + len(fn)) if tp or fn else None,
        "tp_item_ids": tp,
        "fp_item_ids": fp,
        "fn_item_ids": fn,
    }


def candidate_rules(
    voice: dict[str, Any], qwen: dict[str, Any] | None
) -> dict[str, bool]:
    features = voice.get("features") or {}
    qwen_result = (qwen or {}).get("result") or {}
    qwen_visual = bool(qwen) and qwen.get("error") is None and candidate_positive(
        qwen_result,
        include_authority=False,
        allow_unresolved_visual_order=True,
    )
    differs = features.get("speaker_proxy.differs_from_immediate_before") == 1
    new_prior = features.get("speaker_proxy.new_vs_recent_prior") == 1
    sandwich = features.get("speaker_proxy.sandwiched_third_voice") == 1
    return {
        "voice_differs_immediate_before": differs,
        "voice_new_vs_recent_prior": new_prior,
        "voice_sandwich_third_voice": sandwich,
        "qwen_visual_no_order": qwen_visual,
        "qwen_visual_and_voice_differs": qwen_visual and differs,
        "qwen_visual_and_voice_new_prior": qwen_visual and new_prior,
        "qwen_visual_and_voice_sandwich": qwen_visual and sandwich,
    }


def evaluate(
    manual_rows: list[dict[str, Any]],
    voice_rows: list[dict[str, Any]],
    qwen_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manual = {row["item_id"]: row for row in manual_rows}
    if len(manual) != len(manual_rows):
        raise ValueError("duplicate manual item_id")
    qwen = {row["candidate_id"]: row for row in qwen_rows}
    if len(qwen) != len(qwen_rows):
        raise ValueError("duplicate qwen candidate_id")
    audited = []
    rules: dict[str, dict[str, bool]] = {}
    failures = 0
    for voice in voice_rows:
        candidate_id = voice["candidate_id"]
        item_id = voice["item_id"]
        failures += int(bool(voice.get("error")))
        decisions = candidate_rules(voice, qwen.get(candidate_id))
        for name, value in decisions.items():
            rules.setdefault(name, {})[item_id] = rules.setdefault(name, {}).get(item_id, False) or value
        audited.append(
            {
                **voice,
                "manual_reaction_source_role": manual[item_id].get("reaction_source_role"),
                "manual_reaction_content": manual[item_id].get("reaction_content"),
                "manual_description": manual[item_id].get("description"),
                "candidate_rule_decisions": decisions,
            }
        )
    targets = {
        "strict_bystander": {
            item: manual_target(row, include_authority=False)
            for item, row in manual.items()
        },
        "extended_third_party": {
            item: manual_target(row, include_authority=True)
            for item, row in manual.items()
        },
    }
    return audited, {
        "kind": "witnessed_speaker_embedding_proxy_development_audit",
        "status": "failed_or_pending_transfer_no_promotion",
        "policy": "shadow_feature_only_not_diarization_or_acceptance",
        "manual_items": len(manual),
        "candidate_rows": len(voice_rows),
        "candidate_failures": failures,
        "items_with_candidates": len({row["item_id"] for row in voice_rows}),
        "rules": {
            rule: {
                target: metrics(gold, prediction)
                for target, gold in targets.items()
            }
            for rule, prediction in rules.items()
        },
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--voice", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--candidate-ledger", type=Path, required=True)
    parser.add_argument("--audited-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    voice_rows = read_jsonl(args.voice)
    audited, summary = evaluate(
        read_jsonl(args.manual), voice_rows, read_jsonl(args.qwen)
    )
    summary.update(validate_candidate_ledger(voice_rows, read_tsv(args.candidate_ledger)))
    args.audited_output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in audited)
    )
    summary["artifact_sha256"] = {
        "manual": sha256(args.manual),
        "voice": sha256(args.voice),
        "qwen": sha256(args.qwen),
        "candidate_ledger": sha256(args.candidate_ledger),
        "audited_output": sha256(args.audited_output),
    }
    args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
