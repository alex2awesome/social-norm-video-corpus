import json

from experiments.build_reaction_prediction_manifest_v1 import (
    negative_items,
    positive_items,
)
from experiments.selection_function_v1 import analyze, mean


def make_corpus(tmp_path):
    root = tmp_path
    hits = root / "data" / "hits" / "dailymotion__p1"
    hits.mkdir(parents=True)
    (hits / "metadata.json").write_text(json.dumps({
        "provenance": {"scene": {"reaction_strength": 4, "severity": 5,
                                 "n_people": 3, "reactor_role": "bystander",
                                 "scene_type": "action"}}}))
    for uid, tier in (("dailymotion__p1", "no_reaction"), ("dailymotion__n1", "null_verified")):
        d = root / "data" / "negatives" / uid
        d.mkdir(parents=True)
        (d / "neg_0.mp4").write_bytes(b"x")
        (d / "metadata.json").write_text(json.dumps({"modality": tier}))
    cuts = root / "data" / "action_clips_v1"
    cuts.mkdir(parents=True)
    dest = cuts / "clips" / "dailymotion__p1" / "clip_0_action.mp4"
    (cuts / "cut_manifest_shard_0.jsonl").write_text(json.dumps({
        "item_id": "witnessed:dailymotion__p1:clip_0", "uid": "dailymotion__p1",
        "clip_idx": 0, "ok": True, "view": "action", "dest": str(dest)}))
    return root


def test_positive_filtering_and_negative_matching(tmp_path):
    root = make_corpus(tmp_path)
    context = {"dailymotion__p1": {"content_type": "organic_capture",
                                   "norm_domain": "interpersonal_social",
                                   "inferred_norm": "wait your turn"}}
    positives = positive_items(
        sorted(root.glob("data/action_clips_v1/cut_manifest_shard_*.jsonl")),
        {}, context, root)
    assert len(positives) == 1
    p = positives[0]
    assert p["label_reaction_present"] == 1
    assert p["label_reaction_strength"] == 4
    assert p["covariates"]["inferred_norm"] == "wait your turn"
    # News-labeled source is excluded.
    context["dailymotion__p1"]["content_type"] = "news_or_documentary"
    assert positive_items(
        sorted(root.glob("data/action_clips_v1/cut_manifest_shard_*.jsonl")),
        {}, context, root) == []
    negatives, counts = negative_items(root, {"dailymotion__p1"}, max_items=1)
    assert counts["matched"] == 1
    matched = [n for n in negatives if n["covariates"]["matched_same_source"]]
    assert matched[0]["uid"] == "dailymotion__p1"
    assert matched[0]["label_reaction_present"] == 0


def test_selection_function_analysis(tmp_path):
    root = make_corpus(tmp_path)
    sc = tmp_path / "sc.jsonl"
    sc.write_text("\n".join(json.dumps(r) for r in [
        {"uid": "dailymotion__p1", "pillars": ["witnessed"],
         "result": {"norm_domain": "interpersonal_social", "content_type": "organic_capture",
                    "polarity": "violation", "crowd_stance": "no_comments",
                    "inferred_behavior": "x", "inferred_norm": "y",
                    "evidence_fields": ["title"], "note": ""}},
        {"uid": "dailymotion__c1", "pillars": ["commentary"],
         "result": {"norm_domain": "traffic_driving", "content_type": "news_or_documentary",
                    "polarity": "violation", "crowd_stance": "no_comments",
                    "inferred_behavior": "x", "inferred_norm": "y",
                    "evidence_fields": ["title"], "note": ""}},
    ]))
    props = tmp_path / "props.jsonl"
    props.write_text(json.dumps({"uid": "dailymotion__p1", "clip_idx": 0,
                                 "item_id": "witnessed:dailymotion__p1:clip_0"}))
    report = analyze(root, sc, props)
    assert report["witnessed_sources"] == 1
    assert report["mean_reaction_strength_by_severity"]["5"] == {"mean": 4.0, "n": 1}
    assert report["mean_reaction_strength_by_n_people_bucket"]["3-5"]["n"] == 1
    partition = report["norm_domain_partition"]
    assert partition["interpersonal_social"]["witnessed_share"] == 1.0
    assert partition["traffic_driving"]["commentary_share"] == 1.0
    assert mean([]) is None
