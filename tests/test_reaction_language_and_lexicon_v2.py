from scripts.reaction_language_tags_v1 import tag_text
from scripts.witnessed_reaction_text_features import (
    intervention_features,
    intervention_features_v2,
)


def test_script_detection_for_non_latin_reactions():
    assert tag_text("Да чё вы орёте?")["language_guess"] == "script:cyrillic"
    assert tag_text("یہ عدالت ہے کھاڑا نہیں")["language_guess"] == "script:arabic"
    assert tag_text("こんな事するなよ")["dominant_script"] in {"kana", "cjk"}
    assert tag_text("ok")["language_guess"] == "too_short"


def test_latin_stopword_voting():
    assert tag_text("what the hell is that, you can't do this")["language_guess"] == "english"
    assert tag_text("qué está haciendo usted, mira eso hombre")["language_guess"] == "spanish"
    assert tag_text("você não está fazendo isso cara, meu deus")["language_guess"] == "portuguese"
    assert tag_text("qu'est-ce que vous faites, c'est pas possible, arrête")["language_guess"] == "french"
    # Gibberish stays honest.
    assert tag_text("zzzz brglx qwpt mnrv")["language_guess"] == "unknown_latin"


def test_v2_lexicon_catches_gap_phrases_v1_missed():
    for phrase in (
        "hey, that's not fair",
        "how dare you touch her",
        "apologize to him right now",
        "show some respect",
        "you're nothing but a liar",
        "that was so out of line",
    ):
        v1 = intervention_features(phrase)
        v2 = intervention_features_v2(phrase)
        assert v2["intervention.any_active_v2"] == 1.0, phrase
        # These were the misses: v1 either caught them (fine) or v2 rescues.
        assert v2["intervention.any_active"] == v1["intervention.any_active"]


def test_v2_is_a_superset_and_v1_is_untouched():
    text = "stop, that's so rude"
    v1 = intervention_features(text)
    v2 = intervention_features_v2(text)
    for key, value in v1.items():
        assert v2[key] == value
    assert v2["intervention.any_active_v2"] == 1.0
    # Generic affect still is not censure under v2.
    quiet = intervention_features_v2("oh wow omg")
    assert quiet["intervention.any_active_v2"] == 0.0
