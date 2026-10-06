import json

from scripts.compile_commentary_exclusions import compile_exclusions


def test_compiles_source_level_unique_uids(tmp_path):
    base = tmp_path / "base.txt"
    base.write_text(
        "youtube__abc__2__w003\n"
        "commentary:rumble__v1:0:window:2\n"
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "items": [
                    {"uid": "reddit__post"},
                    {"item_id": "commentary:dailymotion__x1:3"},
                    {"uid": "reddit__post"},
                ]
            }
        )
    )

    assert compile_exclusions([base], [manifest]) == [
        "dailymotion__x1",
        "reddit__post",
        "rumble__v1",
        "youtube__abc",
    ]
