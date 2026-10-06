from pathlib import Path

import cv2
import numpy as np

from scripts.score_visual_scene_baselines import (
    completed_item_ids,
    relational_pose_features,
    resolve_clip,
    sample_frames,
    shard_for_item,
    uniformly_subsample_frames,
    xclip_media_kwargs,
    xclip_processor_inputs,
)


def test_resolve_clip_accepts_frozen_audit_source_path(tmp_path: Path):
    manifest = tmp_path / "audit" / "sealed.jsonl"
    manifest.parent.mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    assert resolve_clip({"source_path": str(source)}, manifest) == source


def test_sample_frames_honors_bounded_window(tmp_path: Path):
    target = tmp_path / "test.mp4"
    writer = cv2.VideoWriter(
        str(target),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (32, 32),
    )
    for index in range(100):
        writer.write(np.full((32, 32, 3), index, dtype=np.uint8))
    writer.release()

    frames, media = sample_frames(target, 5, start_sec=2, end_sec=4)
    assert len(frames) == 5
    assert media["sample_start_sec"] == 2
    assert media["sample_end_sec"] == 4
    assert int(frames[0].mean()) < int(frames[-1].mean())


def test_stable_shards_are_disjoint_and_complete():
    item_ids = [f"pillar:uid:{index}" for index in range(100)]
    shards = [
        {item_id for item_id in item_ids if shard_for_item(item_id, 7) == index}
        for index in range(7)
    ]
    assert set.union(*shards) == set(item_ids)
    assert sum(map(len, shards)) == len(item_ids)


def test_completed_item_ids_combines_files_and_ignores_failures(tmp_path: Path):
    first = tmp_path / "base.jsonl"
    second = tmp_path / "shard.jsonl"
    first.write_text(
        '{"item_id":"a","low_level":{"motion_mean":0},"error":null}\n'
        '{"item_id":"b","low_level":{},"error":"decode"}\n'
    )
    second.write_text(
        '{"item_id":"a","low_level":{"motion_mean":0},"error":null}\n'
        '{"item_id":"c","low_level":{"motion_mean":1},"error":null}\n'
    )
    assert completed_item_ids([first, second]) == {"a", "c"}


def test_completed_item_ids_can_require_requested_scorers(tmp_path: Path):
    path = tmp_path / "scores.jsonl"
    path.write_text(
        '{"item_id":"a","low_level":{"motion_mean":0},'
        '"keypoints":{"person_count_mean":1},"clip_scores":{"prompts":["x"]},'
        '"error":null}\n'
        '{"item_id":"b","low_level":{"motion_mean":0},'
        '"keypoints":null,"clip_scores":{"prompts":["x"]},"error":null}\n'
    )
    assert completed_item_ids(
        [path],
        ("low_level", "keypoints", "clip_scores"),
    ) == {"a"}


def test_xclip_prefers_public_video_signature_over_internal_attribute_name():
    class Legacy:
        attributes = ["image_processor", "tokenizer"]

        def __call__(self, text=None, images=None):
            return None

    class Published:
        attributes = ["image_processor", "tokenizer"]

        def __call__(self, text=None, videos=None):
            return None

    class Modern:
        attributes = ["video_processor", "tokenizer"]

    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    assert xclip_media_kwargs(Legacy(), frames) == {"images": frames}
    assert xclip_media_kwargs(Published(), frames) == {"videos": frames}
    assert xclip_media_kwargs(Modern(), frames) == {"videos": frames}


def test_xclip_subsampling_is_uniform_and_includes_endpoints():
    frames = [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(12)]
    sampled = uniformly_subsample_frames(frames, 8)
    assert len(sampled) == 8
    assert int(sampled[0][0, 0, 0]) == 0
    assert int(sampled[-1][0, 0, 0]) == 11


def test_xclip_processor_falls_back_when_video_dispatch_drops_pixels():
    class Processor:
        attributes = ["image_processor", "tokenizer"]

        def __call__(self, text=None, images=None, videos=None, **kwargs):
            if images is not None:
                return {"pixel_values": "pixels", "input_ids": "tokens"}
            return {"input_ids": "tokens"}

    frames = [np.zeros((4, 4, 3), dtype=np.uint8)]
    result = xclip_processor_inputs(Processor(), frames, text=["scene"])
    assert result["pixel_values"] == "pixels"


def test_relational_pose_features_distinguish_stable_single_and_close_pair():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    single = {
        "boxes": np.asarray([[50, 10, 150, 90]], dtype=float),
        "keypoints": np.zeros((1, 17, 3), dtype=float),
    }
    stable = relational_pose_features([single, single], [frame, frame])
    assert stable["stable_single_person_transition_fraction"] == 1.0
    assert stable["close_pair_fraction"] == 0.0

    paired = {
        "boxes": np.asarray(
            [[20, 10, 90, 90], [80, 10, 150, 90]], dtype=float
        ),
        "keypoints": np.zeros((2, 17, 3), dtype=float),
    }
    close = relational_pose_features([paired], [frame])
    assert close["close_pair_fraction"] == 1.0
    assert close["second_to_first_area_ratio_mean"] == 1.0
