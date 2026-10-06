import json
from pathlib import Path

from scripts.sample_instructional_v15_prospective import (
    freeze,
    read_exclusions,
)


def write_source(root: Path, uid: str, demos: list[dict]) -> None:
    source = root / uid
    source.mkdir(parents=True)
    for index, demo in enumerate(demos):
        if demo.get("clip"):
            (source / demo["clip"]).write_bytes(f"clip-{index}".encode())
    (source / "metadata.json").write_text(
        json.dumps({"title": uid, "category": "instr_test", "demos": demos})
    )


def test_freeze_is_platform_specific_deterministic_and_one_per_uid(
    tmp_path: Path,
) -> None:
    root = tmp_path / "instructional"
    write_source(
        root,
        "dailymotion__a",
        [
            {"clip": "d0.mp4", "polarity": "violation", "norm": "a"},
            {"clip": "d1.mp4", "polarity": "violation", "norm": "b"},
        ],
    )
    write_source(
        root,
        "dailymotion__b",
        [{"clip": "d0.mp4", "polarity": "violation", "norm": "c"}],
    )
    write_source(
        root,
        "youtube__ignored",
        [{"clip": "d0.mp4", "polarity": "violation", "norm": "d"}],
    )
    first = freeze(root, set(), "seed", "dailymotion", None)
    second = freeze(root, set(), "seed", "dailymotion", None)
    assert first == second
    assert {row["uid"] for row in first} == {
        "dailymotion__a",
        "dailymotion__b",
    }
    assert all(row["source_platform"] == "dailymotion" for row in first)


def test_read_exclusions_combines_uid_and_source_files(tmp_path: Path) -> None:
    uid_file = tmp_path / "uids.txt"
    source_file = tmp_path / "source.jsonl"
    uid_file.write_text("a\n")
    source_file.write_text(json.dumps({"uid": "b"}) + "\n")
    assert read_exclusions([uid_file], [source_file]) == {"a", "b"}

