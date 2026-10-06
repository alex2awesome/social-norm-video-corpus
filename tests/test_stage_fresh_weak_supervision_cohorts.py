import json

from scripts.stage_fresh_weak_supervision_cohorts import discover_witnessed_inputs


def test_discovery_finds_largest_matching_manifest_and_scores(tmp_path):
    shadow = tmp_path / "data" / "shadow_scores" / "run"
    shadow.mkdir(parents=True)
    manifest = shadow / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "candidate_id": "c", "item_id": "i", "candidate_video_path": "/x.mp4"
    }) + "\n")
    scores = shadow / "scores.jsonl"
    scores.write_text(json.dumps({
        "candidate_id": "c", "result": {}, "prompt_version": "v3"
    }) + "\n")
    found_manifest, found_scores = discover_witnessed_inputs(tmp_path)
    assert found_manifest == manifest
    assert found_scores == scores
