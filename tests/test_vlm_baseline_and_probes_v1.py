import json

import numpy as np
import pytest

from experiments.eval_metrics_v1 import accuracy_at, auroc, sliced_auroc
from experiments.leakage_probes_v1 import featurize, predict, train_logistic
from experiments.run_vlm_baseline_v1 import parse_result, select_pilot


def test_auroc_and_accuracy():
    assert auroc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.1, 0.2, 0.8, 0.9], [1, 1, 0, 0]) == 0.0
    assert auroc([0.5, 0.5], [1, 0]) == 0.5
    assert auroc([0.5], [1]) is None
    acc = accuracy_at([0.9, 0.2], [1, 0], 0.5)
    assert acc["accuracy"] == 1.0 and acc["tp"] == 1 and acc["tn"] == 1
    rows = [{"s": 0.9, "y": 1, "g": "a"}] * 20 + [{"s": 0.1, "y": 0, "g": "a"}] * 20
    assert sliced_auroc(rows, "s", "y", "g")["a"]["auroc"] == 1.0


def test_logistic_probe_learns_separable_text():
    positive = [f"stop shouting rude queue conflict {i}" for i in range(60)]
    negative = [f"calm cooking recipe tutorial pasta {i}" for i in range(60)]
    features = np.stack([featurize(t) for t in positive + negative])
    labels = np.array([1.0] * 60 + [0.0] * 60, dtype=np.float32)
    weights = train_logistic(features, labels)
    scores = predict(weights, features)
    assert auroc([float(s) for s in scores], [1] * 60 + [0] * 60) > 0.95


def test_vlm_contract_validation():
    good = {"people_visible": 3, "reaction_already_visible": False,
            "will_react_probability": 0.8, "predicted_responder": "bystander",
            "predicted_strength": 4, "behavior_seen": "man shoves past a queue"}
    parsed = parse_result("noise " + json.dumps(good) + " tail")
    assert parsed["will_react_probability"] == 0.8
    for corrupt in (
        {**good, "will_react_probability": 1.4},
        {**good, "predicted_responder": "the crowd"},
        {**good, "predicted_strength": 7},
        {**good, "reaction_already_visible": "yes"},
    ):
        with pytest.raises(ValueError):
            parse_result(json.dumps(corrupt))


def test_pilot_selection_balanced_and_deterministic(tmp_path):
    manifest = tmp_path / "m.jsonl"
    rows = []
    for i in range(200):
        kind = "positive" if i % 2 == 0 else "negative"
        rows.append({"item_id": f"i{i}", "kind": kind, "split": "test",
                     "label_reaction_present": int(kind == "positive"),
                     "covariates": {"matched_same_source": i % 4 == 1}})
    rows.append({"item_id": "train_item", "kind": "positive", "split": "train",
                 "label_reaction_present": 1, "covariates": {}})
    manifest.write_text("\n".join(json.dumps(r) for r in rows))
    pilot = select_pilot(manifest, 40)
    assert len(pilot) == 40
    assert sum(1 for r in pilot if r["kind"] == "positive") == 20
    assert sum(1 for r in pilot if r["covariates"].get("matched_same_source")) == 10
    assert all(r["split"] == "test" for r in pilot)
    assert [r["item_id"] for r in pilot] == [r["item_id"] for r in select_pilot(manifest, 40)]
