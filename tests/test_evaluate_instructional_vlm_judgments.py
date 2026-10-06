import csv
import json
from pathlib import Path

from scripts.evaluate_instructional_vlm_judgments import evaluate


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _tsv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_uses_latest_success_and_fails_uncertain_closed(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    visual = tmp_path / "visual.tsv"
    semantic = tmp_path / "semantic.tsv"
    predictions = tmp_path / "predictions.jsonl"
    _jsonl(
        selection,
        [
            {"audit_index": 0, "item_id": "i0"},
            {"audit_index": 1, "item_id": "i1"},
            {"audit_index": 2, "item_id": "i2"},
        ],
    )
    _tsv(
        visual,
        [
            {"audit_index": 0, "visual_demo": "Y"},
            {"audit_index": 1, "visual_demo": "N"},
            {"audit_index": 2, "visual_demo": "Y"},
        ],
    )
    _tsv(
        semantic,
        [
            {"audit_index": 0, "exact_original": "Y"},
            {"audit_index": 1, "exact_original": "N"},
            {"audit_index": 2, "exact_original": "Y"},
        ],
    )
    _jsonl(
        predictions,
        [
            {"item_id": "i0", "model": "m", "error": "failed", "result": None},
            {"item_id": "i0", "model": "m", "error": None, "result": {"keep": "yes"}},
            {"item_id": "i1", "model": "m", "error": None, "result": {"keep": "yes"}},
            {"item_id": "i2", "model": "m", "error": None, "result": {"keep": "uncertain"}},
        ],
    )
    summary = evaluate(
        selection, visual, semantic, predictions, "result.keep", "exact", "m"
    )
    assert summary["fail_closed_confusion"] == {
        "tp": 1,
        "fp": 1,
        "tn": 0,
        "fn": 1,
        "precision": 0.5,
        "recall": 0.5,
        "specificity": 0.0,
        "accuracy": 1 / 3,
    }
    assert summary["uncertain_indices"] == [2]


def test_evaluate_reads_predictions_embedded_in_selection(tmp_path: Path) -> None:
    selection = tmp_path / "selection.jsonl"
    visual = tmp_path / "visual.tsv"
    semantic = tmp_path / "semantic.tsv"
    _jsonl(
        selection,
        [
            {"audit_index": 0, "item_id": "i0", "judge": {"result": {"keep": "yes"}}},
            {"audit_index": 1, "item_id": "i1", "judge": {"result": {"keep": "no"}}},
        ],
    )
    _tsv(
        visual,
        [
            {"audit_index": 0, "visual_demo": "Y"},
            {"audit_index": 1, "visual_demo": "N"},
        ],
    )
    _tsv(
        semantic,
        [
            {"audit_index": 0, "exact_original": "Y"},
            {"audit_index": 1, "exact_original": "N"},
        ],
    )
    summary = evaluate(
        selection,
        visual,
        semantic,
        selection,
        "judge.result.keep",
        "exact",
        embedded=True,
    )
    assert summary["fail_closed_confusion"]["accuracy"] == 1.0
