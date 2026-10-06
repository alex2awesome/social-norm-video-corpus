import json
import subprocess
import sys

from scripts.combine_audited_shadow_scores import combine_file, combine_row
from scripts.weak_signal_registry import load_registry


def registry():
    return load_registry(__import__("pathlib").Path("config/audited_weak_signals_v1.json"))


def test_active_instructional_rank_is_applied_without_disposition():
    result = combine_row(
        {
            "uid": "u",
            "demo_index": 2,
            "instructional_non_explanation_review_priority_v1": "violation",
        },
        registry(),
        "instructional",
    )
    decision = result["audited_combination"]
    assert decision["triggered_rule_ids"] == [
        "instructional_non_explanation_review_priority_v1"
    ]
    assert decision["review_routes"] == ["instructional_demo_manual_review"]
    assert decision["rankings"][0]["priority_tier"] == "standard_above_explanation"
    assert decision["acceptance_label"] is None
    assert decision["corpus_disposition"] is None
    assert decision["delete_media"] is False


def test_failed_transfer_rule_abstains_even_if_input_says_true():
    result = combine_row(
        {"uid": "u", "witnessed_qwen_video_asr_reaction_retrieval_v3": True},
        registry(),
        "witnessed",
    )
    decision = result["audited_combination"]
    assert result["signal_outputs"] == {
        "witnessed_qwen_video_asr_reaction_retrieval_v3": True
    }
    assert decision["triggered_rule_ids"] == []
    assert decision["review_routes"] == []
    assert decision["strict_route_exclusions"] == []


def test_unregistered_exploratory_field_cannot_trigger_a_route():
    result = combine_row(
        {"uid": "u", "promising_new_model": True}, registry(), "witnessed"
    )
    assert result["signal_outputs"] == {}
    assert result["audited_combination"]["triggered_rule_ids"] == []


def test_exact_wwyd_rule_only_excludes_strict_organic_route():
    result = combine_row(
        {"uid": "u", "witnessed_official_wwyd_channel_v1": True},
        registry(),
        "witnessed",
    )
    decision = result["audited_combination"]
    assert decision["strict_route_exclusions"] == ["strict_organic_witnessed"]
    assert decision["review_routes"] == ["instructional_demo_review"]
    assert decision["acceptance_label"] is None


def test_file_combiner_is_one_to_one_append_only(tmp_path):
    source = tmp_path / "scores.jsonl"
    source.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in [
                {
                    "uid": "a",
                    "instructional_non_explanation_review_priority_v1": "correct",
                },
                {
                    "uid": "b",
                    "instructional_non_explanation_review_priority_v1": None,
                    "error": "bad metadata",
                },
            ]
        )
    )
    out = tmp_path / "combined.jsonl"
    summary = combine_file(source, out, registry(), "instructional")
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert summary["rows"] == 2
    assert summary["ranking_rows"] == 1
    assert summary["acceptance_rows"] == 0
    assert summary["input_mutated"] is False
    assert len(rows) == 2
    assert rows[1]["source_error"] == "bad metadata"


def test_direct_script_entry_point_imports_from_repository(tmp_path):
    source = tmp_path / "scores.jsonl"
    source.write_text(json.dumps({"uid": "a"}) + "\n")
    out = tmp_path / "combined.jsonl"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/combine_audited_shadow_scores.py",
            "--input",
            str(source),
            "--out",
            str(out),
            "--registry",
            "config/audited_weak_signals_v1.json",
            "--pillar",
            "instructional",
            "--root",
            ".",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(out.read_text())["input_preserved"] is True


def test_cross_pillar_rule_id_abstains():
    result = combine_row(
        {"uid": "u", "instructional_non_explanation_review_priority_v1": "correct"},
        registry(),
        "commentary",
    )
    assert result["signal_outputs"] == {}
    assert result["audited_combination"]["triggered_rule_ids"] == []
