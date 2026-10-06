from scripts.export_commentary_visual_recovery_audit import resolve_media


def test_resolve_media_accepts_stable_retained_video_name(tmp_path):
    media = tmp_path / "dailymotion__x.mp4"
    media.write_bytes(b"video")
    (tmp_path / "dailymotion__x.info.json").write_text("{}")

    assert resolve_media(tmp_path, 3, "dailymotion__x") == media


def test_resolve_media_prefers_legacy_prefixed_qa_name(tmp_path):
    prefixed = tmp_path / "03_dailymotion__x.mp4"
    direct = tmp_path / "dailymotion__x.mp4"
    prefixed.write_bytes(b"video")
    direct.write_bytes(b"video")

    assert resolve_media(tmp_path, 3, "dailymotion__x") == prefixed
