from weaksup.norm_category_tags_v1 import categorize_item, categorize_norm


def test_affect_labels_are_not_norms():
    for norm in ("surprise", "concern", "confusion", "frustration", "shock"):
        assert categorize_norm(norm) == "affect_only", norm


def test_social_norms_map_to_interpersonal():
    for norm in ("respect", "personal space", "honesty", "fairness",
                 "apology", "consent", "respect for others", "civility",
                 "queue etiquette", "harassment"):
        assert categorize_norm(norm) == "interpersonal_social", norm


def test_other_domains():
    assert categorize_norm("safe driving") == "traffic_driving"
    assert categorize_norm("reckless driving") == "traffic_driving"
    assert categorize_norm("aggression") == "violence_conflict"
    assert categorize_norm("non-violence") == "violence_conflict"
    assert categorize_norm("public safety") == "safety_physical"
    assert categorize_norm("respect for authority") == "interpersonal_social"
    assert categorize_norm("authority") == "authority_legal"
    assert categorize_norm("respect for property") == "interpersonal_social"
    assert categorize_norm("theft") == "property"
    assert categorize_norm("") == "missing"
    assert categorize_norm("zorble") == "other_unmapped"


def test_item_rollup_prefers_social_category():
    item = categorize_item([{"norm": "surprise"}, {"norm": "personal space"}])
    assert item["primary_category"] == "interpersonal_social"
    assert item["any_social_norm_category"] is True
    assert item["affect_only_item"] is False
    pure_affect = categorize_item([{"norm": "surprise"}, {"norm": "confusion"}])
    assert pure_affect["affect_only_item"] is True
    assert pure_affect["any_social_norm_category"] is False
