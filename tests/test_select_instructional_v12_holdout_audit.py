import json
from pathlib import Path

from scripts.select_instructional_v12_holdout_audit import select


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _metadata(item_id: str, strict: bool) -> dict:
    return {
        "item_id": item_id,
        "error": None,
        "result": {
            "exact_violation_demo": "yes" if strict else "no",
            "performed_violation_not_only_described": (
                "yes" if strict else "no"
            ),
            "proposed_actor_behavior_target_match": (
                "yes" if strict else "no"
            ),
            "event_polarity": "violation" if strict else "neutral",
        },
    }


def test_v12_selector_excludes_discovery_and_deduplicates_controls(
    tmp_path: Path,
) -> None:
    source = []
    storyboards = []
    v12 = []
    qwen = []
    gemma = []
    for index in range(12):
        item_id = f"i{index}"
        source.append({"item_id": item_id, "uid": f"u{index}"})
        storyboards.append(
            {
                "item_id": item_id,
                "sheet_path": f"/tmp/{item_id}.jpg",
                "sheet_sha256": f"h{index}",
            }
        )
        v12.append(
            {
                "item_id": item_id,
                "error": None,
                "result": {
                    "strict_exact_alignment": (
                        "yes" if index in {0, 1} else "no"
                    )
                },
            }
        )
        qwen.append(_metadata(item_id, index in {2, 3, 4}))
        gemma.append(_metadata(item_id, index in {3, 4, 5}))
    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in (
            "source",
            "storyboards",
            "v12",
            "qwen",
            "gemma",
            "exclude1",
            "exclude2",
        )
    }
    for name, rows in (
        ("source", source),
        ("storyboards", storyboards),
        ("v12", v12),
        ("qwen", qwen),
        ("gemma", gemma),
        ("exclude1", [{"uid": "u11"}]),
        ("exclude2", [{"uid": "old"}]),
    ):
        _write(paths[name], rows)

    semantic, blind, summary = select(
        paths["source"],
        paths["storyboards"],
        paths["v12"],
        paths["qwen"],
        paths["gemma"],
        [paths["exclude1"], paths["exclude2"]],
        candidate_cap=60,
        control_per_stratum=2,
        blind_seed="seed",
    )

    assert summary["excluded_discovery_uids"] == 2
    assert summary["candidate_population_items"] == 2
    assert summary["control_items"] == 6
    assert summary["control_counts"] == {
        "control_gemma_exact_v12_reject": 2,
        "control_qwen_exact_v12_reject": 2,
        "control_remaining_v12_reject": 2,
    }
    assert len({row["item_id"] for row in semantic}) == 8
    assert len(blind) == 8
    assert all(
        set(row)
        == {"audit_index", "candidate_id", "sheet_path", "sheet_sha256"}
        for row in blind
    )
