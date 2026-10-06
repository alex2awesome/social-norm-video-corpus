from scripts.sample_commentary_visual_audit import (
    diversity_sample,
    load_exclusions,
    load_manifest_exclusions,
    normalize_source_uid,
    select,
)


def row(item_id, platform, query, category, signal):
    return {
        "item_id": item_id,
        "source_platform": platform,
        "query_source": query,
        "category": category,
        "signal": signal,
    }


def test_platform_quotas_and_seed_are_deterministic():
    rows = [
        row("a", "dailymotion", "taxonomy", "one", "criticism"),
        row("b", "dailymotion", "expand", "two", "rule"),
        row("c", "dailymotion", "taxonomy", "three", "rule"),
        row("d", "youtube", "taxonomy", "one", "criticism"),
        row("e", "youtube", "expand", "two", "rule"),
    ]

    first, availability = select(
        rows, {"dailymotion": 2, "youtube": 1}, "seed"
    )
    second, _ = select(rows, {"dailymotion": 2, "youtube": 1}, "seed")

    assert [item["item_id"] for item in first] == [
        item["item_id"] for item in second
    ]
    assert availability == {"dailymotion": 3, "youtube": 2}
    assert sum(item["source_platform"] == "dailymotion" for item in first) == 2
    assert sum(item["source_platform"] == "youtube" for item in first) == 1


def test_diversity_round_robin_uses_different_groups_before_repeats():
    rows = [
        row("a1", "dailymotion", "taxonomy", "one", "rule"),
        row("a2", "dailymotion", "taxonomy", "one", "rule"),
        row("b1", "dailymotion", "expand", "two", "criticism"),
    ]

    selected = diversity_sample(rows, 2, "seed")

    assert len(
        {
            (item["query_source"], item["category"], item["signal"])
            for item in selected
        }
    ) == 2


def test_source_uid_normalizes_microclip_and_item_identifiers(tmp_path):
    assert (
        normalize_source_uid("youtube__abc_DEF-123__3__w006")
        == "youtube__abc_DEF-123"
    )
    assert (
        normalize_source_uid("commentary:rumble__v123:2:window:4")
        == "rumble__v123"
    )
    exclusions = tmp_path / "uids.txt"
    exclusions.write_text(
        "youtube__abc_DEF-123__3__w006\n"
        "commentary:rumble__v123:2:window:4\n"
    )
    assert load_exclusions(exclusions) == {
        "youtube__abc_DEF-123",
        "rumble__v123",
    }


def test_manifest_exclusions_accept_json_and_jsonl(tmp_path):
    json_manifest = tmp_path / "manifest.json"
    json_manifest.write_text(
        '{"items":[{"uid":"dailymotion__x1"},'
        '{"item_id":"commentary:youtube__abc:2"}]}'
    )
    assert load_manifest_exclusions(json_manifest) == {
        "dailymotion__x1",
        "youtube__abc",
    }

    jsonl_manifest = tmp_path / "manifest.jsonl"
    jsonl_manifest.write_text(
        '{"uid":"reddit__one"}\n'
        '{"item_id":"commentary:rumble__v2:1:window:0"}\n'
    )
    assert load_manifest_exclusions(jsonl_manifest) == {
        "reddit__one",
        "rumble__v2",
    }
