import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_commentary_consensus_clip_plan.py"
SPEC = importlib.util.spec_from_file_location("commentary_clip_plan", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_shorter_interval_is_expanded_and_clamped():
    qwen = {"result": {"candidate_start_sec": 10, "candidate_end_sec": 30}}
    gemma = {"result": {"candidate_start_sec": 1, "candidate_end_sec": 2}}
    start, end, proposer = MODULE.proposed_bounds(qwen, gemma, 20)
    assert proposer == "gemma"
    assert start == 0
    assert end == 6


def test_long_interval_is_capped():
    qwen = {"result": {"candidate_start_sec": 10, "candidate_end_sec": 40}}
    gemma = {"result": {"candidate_start_sec": 20, "candidate_end_sec": 35}}
    start, end, proposer = MODULE.proposed_bounds(qwen, gemma, 60)
    assert proposer == "gemma"
    assert end - start == 12
