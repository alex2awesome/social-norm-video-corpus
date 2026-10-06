import subprocess
import sys
from pathlib import Path

from scripts.evaluate_commentary_microclips import consecutive_run, evaluate_microclips


def result(value):
    answer = "yes" if value else "no"
    return {
        "situated_social_scenario_visible": answer,
        "concrete_action_or_situated_speech_visible": answer,
        "proposed_norm_plausibly_demonstrated": answer,
    }


def test_consecutive_run():
    assert consecutive_run(set()) == 0
    assert consecutive_run({0, 1, 3, 4, 5}) == 3


def test_dual_same_window_differs_from_dual_parent_evidence():
    manual = [
        {
            "item_id": "parent",
            "expected_disposition": "dense_followup_commentary",
            "demo_quality": "clear_visual",
        }
    ]
    windows = [
        {"item_id": "w0", "parent_item_id": "parent", "window_index": 0},
        {"item_id": "w1", "parent_item_id": "parent", "window_index": 1},
    ]
    primary = [
        {"item_id": "w0", "result": result(True)},
        {"item_id": "w1", "result": result(False)},
    ]
    secondary = [
        {"item_id": "w0", "result": result(False)},
        {"item_id": "w1", "result": result(True)},
    ]

    report = evaluate_microclips(manual, windows, primary, secondary)

    same = report["rules"]["dual_same_window_at_least_1"]["strict_clear_targets"]
    parent = report["rules"]["dual_parent_each_at_least_1"]["strict_clear_targets"]
    assert same["predicted_positive"] == 0
    assert parent["predicted_positive"] == 1


def test_direct_script_entrypoint_loads_sibling_module():
    repo_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "evaluate_commentary_microclips.py"),
            "--help",
        ],
        cwd=repo_root,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--primary" in result.stdout
