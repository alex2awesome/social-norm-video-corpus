from scripts.select_witnessed_role_causal_binding_v6_fresh import (
    reaction_outcome,
    select,
)


def candidate(uid, index=0):
    return {
        "candidate_id": f"witnessed:{uid}:clip_0:candidate_{index}",
        "item_id": f"witnessed:{uid}:clip_0", "uid": uid,
        "candidate_video_path": f"/media/{uid}_{index}.mp4",
    }


def score(row, staging="clearly_staged", role="separate_bystander"):
    return {
        "candidate_id": row["candidate_id"], "error": None,
        "result": {
            "reaction_grounded": "yes", "action_before_or_overlaps_response": "yes",
            "response_targets_action": "yes", "responder_role": role,
            "response_content": "targeted_objection", "trigger_kind": "interpersonal_treatment",
            "staging": staging,
        },
    }


def test_enrichment_target_deliberately_ignores_staging():
    row = candidate("dailymotion__a")
    assert reaction_outcome([row], {row["candidate_id"]: score(row)}) == "v3_staging_independent_reaction_positive"


def test_selection_is_source_disjoint_with_separate_uniform_and_enrichment():
    manifest = []
    scores = []
    for index in range(12):
        uid = f"dailymotion__{index}"
        row = candidate(uid)
        manifest.append(row)
        scores.append(score(row, role=("affected_target" if index < 3 else "separate_bystander")))
    selected = select(manifest, scores, {"dailymotion__11"}, uniform=3, enriched=4, seed="seed")
    assert len(selected) == 7
    assert len({row["uid"] for row in selected}) == 7
    assert {row["cohort"] for row in selected} == {"uniform_probability_sample", "v3_positive_enrichment"}
    assert "dailymotion__11" not in {row["uid"] for row in selected}
