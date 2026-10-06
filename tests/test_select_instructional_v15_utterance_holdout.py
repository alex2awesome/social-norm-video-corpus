from scripts.select_instructional_v15_utterance_holdout import select, stage


def visual(
    item_id: str,
    *,
    strict: bool = True,
    evidence_source: str = "physical_action",
    quote: bool = False,
) -> dict:
    result = {
        "observable_event": "yes",
        "event_start_percent": 10,
        "event_end_percent": 40,
        "literal_actor": "person",
        "literal_action_or_situated_utterance": (
            "says 'stop doing that'" if quote else "interrupts"
        ),
        "literal_affected_party_or_shared_setting": "other person",
        "same_event_actor_action_target": "yes",
        "event_temporally_localized": "yes",
        "evidence_source": evidence_source,
        "scene_role": "demonstrated_event",
        "demonstration_kind": "interpersonal_conduct",
    }
    if not strict:
        result["observable_event"] = "no"
    return {"item_id": item_id, "result": result}


def verifier(
    item_id: str,
    *,
    usable: str = "yes",
    response: str = "yes",
) -> dict:
    return {
        "item_id": item_id,
        "result": {
            "demo_usable": usable,
            "social_response_or_consequence_present": response,
        },
    }


def test_stage_adds_utterance_anchor_after_v14_checks() -> None:
    assert stage(visual("a"), verifier("a")) == "candidate"
    assert stage(
        visual("a", evidence_source="situated_dialogue_or_subtitles"),
        verifier("a"),
    ) == "utterance_anchor_reject"
    assert stage(
        visual(
            "a",
            evidence_source="situated_dialogue_or_subtitles",
            quote=True,
        ),
        verifier("a"),
    ) == "candidate"
    assert stage(visual("a"), verifier("a", response="no")) == "response_reject"


def test_select_blinds_semantics_and_samples_each_control_band() -> None:
    sources = [
        {
            "item_id": f"i{i}",
            "uid": f"dailymotion__{i}",
            "source_platform": "dailymotion",
            "polarity": "violation",
            "norm": f"secret-{i}",
        }
        for i in range(8)
    ]
    storyboards = {
        f"i{i}": {
            "item_id": f"i{i}",
            "sheet_path": f"/s/{i}.jpg",
            "sheet_sha256": str(i),
        }
        for i in range(8)
    }
    visuals = {f"i{i}": visual(f"i{i}") for i in range(8)}
    verifiers = {f"i{i}": verifier(f"i{i}") for i in range(8)}
    visuals["i2"] = visual(
        "i2", evidence_source="situated_dialogue_or_subtitles"
    )
    visuals["i3"] = visual(
        "i3", evidence_source="situated_dialogue_or_subtitles"
    )
    verifiers["i4"] = verifier("i4", response="no")
    verifiers["i5"] = verifier("i5", response="no")
    verifiers["i6"] = verifier("i6", usable="no")
    verifiers["i7"] = verifier("i7", usable="no")

    semantic, blind, summary = select(
        sources,
        storyboards,
        visuals,
        verifiers,
        controls_per_band=1,
    )
    assert summary["candidate_rows"] == 2
    assert summary["control_counts"] == {
        "response_reject": 1,
        "utterance_anchor_reject": 1,
        "v10a_reject": 1,
    }
    assert len(semantic) == len(blind) == 5
    assert all("norm" not in row and "item_id" not in row for row in blind)

