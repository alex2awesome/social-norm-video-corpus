from scripts.render_instructional_v9_validation_pages import panel_label


def test_panel_label_is_opaque():
    row = {
        "audit_index": 7,
        "candidate_id": "v9fresh-0007",
        "norm": "secret",
        "polarity": "violation",
    }
    label = panel_label(row)
    assert label == "#0007  v9fresh-0007"
    assert "secret" not in label
    assert "violation" not in label
