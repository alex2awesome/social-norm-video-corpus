from scripts.export_commentary_unlabeled_benchmark import (
    audio_output_args,
    requested_duration_flags,
    seek_input_args,
)


def test_commentary_proxy_audio_is_preserved_unless_explicitly_stripped():
    assert audio_output_args(False) == ["-c:a", "aac", "-b:a", "96k"]
    assert audio_output_args(True) == ["-an"]


def test_commentary_proxy_seek_modes_include_bounded_accurate_fallback(tmp_path):
    source = tmp_path / "source.mp4"
    assert seek_input_args(source, 501.742, "fast") == [
        "-ss",
        "501.742",
        "-i",
        str(source),
    ]
    assert seek_input_args(source, 501.742, "hybrid_30s") == [
        "-ss",
        "471.742",
        "-i",
        str(source),
        "-ss",
        "30.000",
    ]
    assert seek_input_args(source, 1.548, "accurate") == [
        "-i",
        str(source),
        "-ss",
        "1.548",
    ]


def test_requested_duration_contract_rejects_one_frame_false_success():
    timing = {"video_duration_sec": 1 / 3}
    assert requested_duration_flags(timing, 6.0) == [
        "requested_duration_coverage=0.055556"
    ]


def test_requested_duration_contract_accepts_small_codec_rounding_loss():
    assert requested_duration_flags({"video_duration_sec": 5.8}, 6.0) == []
