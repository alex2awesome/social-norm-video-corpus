import json
from pathlib import Path

from scripts.report_cached_audiovisual_models import scan


def test_scan_finds_audio_architecture_without_loading_or_network(tmp_path: Path):
    snapshot = (
        tmp_path / "hub/models--Qwen--Qwen2.5-Omni-7B/snapshots/revision"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({
        "model_type": "qwen2_5_omni",
        "architectures": ["Qwen2_5OmniForConditionalGeneration"],
        "audio_config": {"model_type": "whisper"},
    }))
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"")
    report = scan([tmp_path])
    assert report["candidate_count"] == 1
    candidate = report["candidate_snapshots"][0]
    assert candidate["repo_id"] == "Qwen/Qwen2.5-Omni-7B"
    assert candidate["weight_files_present"] is True
    assert "omni" in candidate["audio_capability_markers"]
    assert report["network_accessed"] is False
    assert report["models_loaded"] is False


def test_scan_ignores_ordinary_vision_language_cache(tmp_path: Path):
    snapshot = tmp_path / "models--Qwen--Qwen3-VL-8B/snapshots/revision"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({
        "model_type": "qwen3_vl",
        "architectures": ["Qwen3VLForConditionalGeneration"],
    }))
    report = scan([tmp_path])
    assert report["candidate_count"] == 0
