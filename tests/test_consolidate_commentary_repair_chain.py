import copy
from pathlib import Path

import pytest

from scripts.consolidate_commentary_repair_chain import consolidate, read_jsonl, read_tsv


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "audit_runs" / "20260805_commentary_temporal_repairs_v2"


def real_chain():
    return (
        read_jsonl(RUN / "manifest.jsonl"),
        read_tsv(RUN / "manual_ledger.tsv"),
        [
            (
                read_jsonl(RUN / "repair_plan_v2.jsonl"),
                read_jsonl(RUN / "repair_materialized_v3_fullframes" / "manifest.jsonl"),
                read_tsv(RUN / "repair_materialized_v3_fullframes" / "manual_ledger.tsv"),
            ),
            (
                read_jsonl(RUN / "repair_plan_v3.jsonl"),
                read_jsonl(RUN / "repair_materialized_v4_final" / "manifest.jsonl"),
                read_tsv(RUN / "repair_materialized_v4_final" / "manual_ledger.tsv"),
            ),
        ],
    )


def test_real_repair_chain_resolves_five_of_eight_visual_candidates():
    rows, summary = consolidate(*real_chain())
    assert summary["initial_candidates"] == 8
    assert summary["resolved_candidates"] == 8
    assert summary["passed_visual_candidate_review"] == 5
    assert summary["failed_visual_candidate_review"] == 3
    assert summary["unresolved_repairs"] == 0
    assert summary["pass_rate"] == 0.625
    assert all(row["acceptance_label"] is None for row in rows)
    assert all(row["corpus_disposition"] is None for row in rows)
    assert all(row["delete_media"] is False for row in rows)
    assert {
        row["audit_index"]
        for row in rows
        if row["final_outcome"].startswith("pass_")
    } == {32, 50, 79, 92, 138}


def test_unresolved_repair_fails_closed():
    manifest, ledger, stages = real_chain()
    with pytest.raises(ValueError, match="unresolved repair"):
        consolidate(manifest, ledger, stages[:1])


def test_pass_without_all_visual_gates_is_rejected():
    manifest, ledger, stages = real_chain()
    broken = copy.deepcopy(stages)
    broken[1][2][1]["action_visible"] = "no"
    with pytest.raises(ValueError, match="does not satisfy all visual gates"):
        consolidate(manifest, ledger, broken)
