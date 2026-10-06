import json
from pathlib import Path

from scripts.select_instructional_v11_holdout_audit import select


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _vlm(item_id: str, uid: str, strict: bool) -> dict:
    return {
        "item_id": item_id,
        "uid": uid,
        "model": "m",
        "error": None,
        "result": {
            "exact_violation_demo": "yes" if strict else "no",
            "performed_violation_not_only_described": "yes" if strict else "no",
            "proposed_actor_behavior_target_match": "yes" if strict else "no",
            "event_polarity": "violation" if strict else "neutral",
        },
    }


def test_select_excludes_discovery_and_stratifies_controls(tmp_path: Path) -> None:
    source = []
    storyboards = []
    qwen = []
    glm = []
    consensus = []
    patterns = [
        (True, True, True),
        (True, False, False),
        (False, True, False),
        (True, True, False),
        (False, False, False),
    ]
    for index, (q, g, c) in enumerate(patterns):
        item_id, uid = f"i{index}", f"u{index}"
        source.append({"item_id": item_id, "uid": uid})
        storyboards.append(
            {
                "item_id": item_id,
                "uid": uid,
                "sheet_path": f"/tmp/{item_id}.jpg",
                "sheet_sha256": f"h{index}",
            }
        )
        qwen.append(_vlm(item_id, uid, q))
        glm.append(_vlm(item_id, uid, g))
        consensus.append(
            {
                "item_id": item_id,
                "uid": uid,
                "model": "c",
                "error": None,
                "result": {
                    "strict_exact_violation": "yes" if c else "no"
                },
            }
        )
    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in (
            "source",
            "storyboards",
            "qwen",
            "glm",
            "consensus",
            "discovery",
        )
    }
    for name, rows in (
        ("source", source),
        ("storyboards", storyboards),
        ("qwen", qwen),
        ("glm", glm),
        ("consensus", consensus),
        ("discovery", [{"item_id": "old", "uid": "old_uid"}]),
    ):
        _write(paths[name], rows)

    semantic, blind, summary = select(
        paths["source"],
        paths["storyboards"],
        paths["qwen"],
        paths["glm"],
        paths["consensus"],
        paths["discovery"],
        candidate_cap=60,
        control_count=3,
        blind_seed="seed",
    )

    assert summary["candidate_population_items"] == 1
    assert summary["control_stage_counts"] == {
        "both_exact_consensus_reject": 1,
        "glm_only_exact": 1,
        "qwen_only_exact": 1,
    }
    assert len(semantic) == len(blind) == 4
    assert {row["band"] for row in semantic} == {
        "primary_candidate",
        "control_reject",
    }
    assert all("uid" not in row for row in blind)
