import json
from pathlib import Path

from scripts.sample_instructional_v10_prospective import sample


def write_source(root: Path, uid: str, demos: list[dict]):
    source = root / uid
    source.mkdir(parents=True)
    for index, demo in enumerate(demos):
        if demo.get("clip"):
            (source / demo["clip"]).write_bytes(f"clip-{index}".encode())
    (source / "metadata.json").write_text(
        json.dumps(
            {
                "title": uid,
                "category": "instr_test",
                "demos": demos,
            }
        )
    )


def test_sample_is_deterministic_one_per_uid_and_excludes_sources(tmp_path):
    root = tmp_path / "instructional"
    write_source(
        root,
        "youtube__a",
        [
            {"clip": "demo_0.mp4", "polarity": "violation", "norm": "a"},
            {"clip": "demo_1.mp4", "polarity": "violation", "norm": "b"},
        ],
    )
    write_source(
        root,
        "youtube__b",
        [{"clip": "demo_0.mp4", "polarity": "violation", "norm": "c"}],
    )
    write_source(
        root,
        "youtube__excluded",
        [{"clip": "demo_0.mp4", "polarity": "violation", "norm": "d"}],
    )
    write_source(
        root,
        "dailymotion__ignored",
        [{"clip": "demo_0.mp4", "polarity": "violation", "norm": "e"}],
    )
    exclusion = tmp_path / "excluded.txt"
    exclusion.write_text("youtube__excluded\n")
    first = sample(root, exclusion, "seed", 2)
    second = sample(root, exclusion, "seed", 2)
    assert first == second
    assert {row["uid"] for row in first} == {"youtube__a", "youtube__b"}
    assert len({row["uid"] for row in first}) == 2
    assert all(row["polarity"] == "violation" for row in first)
