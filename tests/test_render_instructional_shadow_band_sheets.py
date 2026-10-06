from scripts.render_instructional_shadow_band_sheets import shorten


def test_shorten_normalizes_and_bounds_labels():
    assert shorten(" a   b ", 10) == "a b"
    assert shorten("abcdefgh", 5) == "abcd…"
