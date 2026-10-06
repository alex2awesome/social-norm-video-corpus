import json
from pathlib import Path

from scripts.select_instructional_v10_prospective_audit import select


def dump(path: Path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def visual(item, result):
    return {
        "item_id": item,
        "model": "model",
        "result": result,
        "error": None,
    }


def test_selects_all_candidates_and_deterministic_controls(tmp_path):
    source = []
    storyboards = []
    v9a = []
    v9b = []
    v10a = []
    v10c = []
    for index in range(8):
        item = f"instructional:youtube__{index}:{index}"
        source.append(
            {
                "item_id": item,
                "uid": f"youtube__{index}",
                "source_platform": "youtube",
                "polarity": "violation",
            }
        )
        storyboards.append(
            {
                "item_id": item,
                "sheet_path": f"/sheet/{index}.jpg",
                "sheet_sha256": str(index),
            }
        )
        v9a.append(
            visual(
                item,
                {
                    "observable_event": "yes" if index < 5 else "no",
                    "same_event_actor_action_target": "yes",
                    "event_temporally_localized": "yes",
                    "scene_role": "demonstrated_event",
                },
            )
        )
        v9b.append(visual(item, {"usable_after_relabel": "yes"}))
        v10a.append(visual(item, {"demo_usable": "yes"}))
        v10c.append(
            visual(
                item,
                {"strict_exact_candidate": "yes" if index < 3 else "no"},
            )
        )
    paths = {}
    for name, rows in [
        ("source", source),
        ("storyboards", storyboards),
        ("v9a", v9a),
        ("v9b", v9b),
        ("v10a", v10a),
        ("v10c", v10c),
    ]:
        paths[name] = tmp_path / f"{name}.jsonl"
        dump(paths[name], rows)
    semantic, blind, summary = select(
        paths["source"],
        paths["storyboards"],
        paths["v9a"],
        paths["v9b"],
        paths["v10a"],
        paths["v10c"],
        control_count=2,
        control_seed="control",
        blind_seed="blind",
    )
    assert summary["candidate_items"] == 3
    assert summary["control_items"] == 2
    assert len(semantic) == len(blind) == 5
    assert sum(row["band"] == "primary_candidate" for row in semantic) == 3
    assert all(set(row) == {"audit_index", "candidate_id", "sheet_path", "sheet_sha256"} for row in blind)
