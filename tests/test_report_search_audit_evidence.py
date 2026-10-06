import json
from pathlib import Path

from scripts.report_search_audit_evidence import canonical_latest, summarize


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_consolidates_manual_artifacts_without_treating_query_as_label(tmp_path: Path):
    run = tmp_path / "audit_runs" / "one"
    run.mkdir(parents=True)
    manifest = {
        "pillar": "instructional",
        "batch_id": "b1",
        "selection_strategy": "balanced_failure_enriched",
        "items": [
            {
                "item_id": "instructional:u:0",
                "uid": "u",
                "pillar": "instructional",
                "query_source": "instructional",
                "found_by_query": "role play",
                "category": "instr_test",
            }
        ],
    }
    (run / "manifest.json").write_text(json.dumps(manifest))
    write_jsonl(
        run / "manual_results.jsonl",
        [
            {
                "item_id": "instructional:u:0",
                "batch_id": "b1",
                "rubric_version": "instructional_v4",
                "model": "manual",
                "pass_index": 0,
                "audited_at": 1,
                "is_social_norm": "yes",
                "visual_demo_present": "no",
                "norm_supported": "yes",
                "decision": "reject",
            }
        ],
    )
    report = summarize(tmp_path / "audit_runs")
    pillar = report["pillars"]["instructional"]
    assert pillar["unique_source_items"] == 1
    assert pillar["axis_counts"]["strict_keep"] == 0
    assert pillar["by_found_by_query"][0]["value"] == "role play"
    assert pillar["by_found_by_query"][0]["rates"]["strict_keep"] == 0
    assert report["interpretation"]["query_or_title_is_label"] is False


def test_latest_pass_wins_without_mixing_pillars():
    rows = [
        {
            "pillar": "instructional",
            "item_id": "same",
            "pass_index": 0,
            "audited_at": 2,
            "results": "a",
        },
        {
            "pillar": "instructional",
            "item_id": "same",
            "pass_index": 1,
            "audited_at": 1,
            "results": "b",
        },
        {
            "pillar": "commentary",
            "item_id": "same",
            "pass_index": 0,
            "audited_at": 3,
            "results": "c",
        },
    ]
    latest = canonical_latest(rows)
    assert len(latest) == 2
    selected = {(row["pillar"], row["results"]) for row in latest}
    assert selected == {("instructional", "b"), ("commentary", "c")}


def test_repair_cannot_inflate_source_query_yield(tmp_path: Path):
    root = tmp_path / "audit_runs"
    source = root / "source"
    repair = root / "repair"
    source.mkdir(parents=True)
    repair.mkdir(parents=True)
    item = {
        "item_id": "instructional:u:0",
        "uid": "u",
        "pillar": "instructional",
        "query_source": "instructional",
        "found_by_query": "demo query",
        "category": "instr_test",
    }
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "pillar": "instructional",
                "batch_id": "source_1",
                "rubric_version": "instructional_v4",
                "items": [item],
            }
        )
    )
    repaired_item = {
        **item,
        "source_frame_manifest_sha256": "source-hash",
        "repair_type": "tighten_to_demo",
    }
    (repair / "manifest.json").write_text(
        json.dumps(
            {
                "pillar": "instructional",
                "batch_id": "recut_1",
                "rubric_version": "instructional_recut_v1",
                "items": [repaired_item],
            }
        )
    )
    rejected = {
        "item_id": item["item_id"],
        "is_social_norm": "yes",
        "visual_demo_present": "no",
        "norm_supported": "yes",
        "decision": "reject",
    }
    accepted = {
        "item_id": item["item_id"],
        "is_social_norm": "yes",
        "visual_demo_present": "yes",
        "norm_supported": "yes",
        "decision": "accept",
    }
    write_jsonl(source / "manual_results.jsonl", [rejected])
    write_jsonl(repair / "manual_results.jsonl", [accepted])
    report = summarize(root)
    pillar = report["pillars"]["instructional"]
    assert pillar["axis_counts"]["strict_keep"] == 0
    assert pillar["repair_followups"]["strict_keep"] == 1
