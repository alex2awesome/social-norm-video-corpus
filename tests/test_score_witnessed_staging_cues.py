from scripts.score_witnessed_staging_cues import score_text, transcript_text


def test_title_prank_is_explicit_staging_candidate():
    row = score_text("Phone snatching prank", "A target objects.")
    assert row["title_staging_cue"] is True
    assert row["explicit_staging_candidate"] is True
    assert row["witnessed_creator_staging_title_v2"] is True


def test_explicit_reveal_is_detected():
    row = score_text(None, "Relax, we're just doing a joke. You're on camera.")
    assert row["transcript_explicit_reveal_cue"] is True
    assert row["explicit_staging_candidate"] is True


def test_creator_setup_requires_intro_and_plan_or_call_to_action():
    assert score_text(None, "What's up guys? Today we're going to test strangers.")[
        "transcript_creator_setup_cue"
    ]
    assert not score_text(None, "What's up guys? Here is the news.")[
        "transcript_creator_setup_cue"
    ]


def test_ordinary_use_of_joke_does_not_trigger():
    row = score_text("Comedy discussion", "That joke was funny.")
    assert row["explicit_staging_candidate"] is False


def test_transcript_text_flattens_segments():
    assert transcript_text({"segments": [{"text": "one"}, {"text": "two"}]}) == "one two"


def test_creator_initiated_title_is_candidate_only():
    row = score_text("Trying to Kiss Girls in the Library", "")
    assert row["title_creator_initiated_candidate_cue"] is True
    assert row["title_staging_cue"] is False
    assert row["explicit_staging_candidate"] is False
    assert row["audited_strict_organic_exclusion_candidate"] is True
    assert row["witnessed_creator_staging_title_v2"] is True


def test_legitimate_instruction_does_not_match_creator_initiated_candidate():
    row = score_text("How to hug a grieving friend", "")
    assert row["title_creator_initiated_candidate_cue"] is False


def test_descriptive_touching_title_is_not_creator_initiated_evidence():
    row = score_text("Indian man touching women without consent", "")
    assert row["title_creator_initiated_candidate_cue"] is False


def test_explicit_attempt_to_touch_remains_creator_initiated_candidate():
    row = score_text("Trying to touch strangers in public", "")
    assert row["title_creator_initiated_candidate_cue"] is True


def test_wwyd_is_candidate_only_before_visual_confirmation():
    row = score_text("What Would You Do? | WWYD", "")
    assert row["title_wwyd_candidate_cue"] is True
    assert row["title_staging_cue"] is False
    assert row["audited_strict_organic_exclusion_candidate"] is False


def test_exact_official_wwyd_channel_is_audited_organic_exclusion():
    row = score_text(
        "A staged restaurant scenario",
        "",
        channel="  WHAT WOULD YOU DO? ",
    )
    assert row["official_wwyd_channel_cue"] is True
    assert row["witnessed_official_wwyd_channel_v1"] is True
    assert row["audited_strict_organic_exclusion_candidate"] is True
    assert row["explicit_staging_candidate"] is True


def test_wwyd_title_or_lookalike_channel_does_not_gain_official_status():
    row = score_text(
        "WWYD if this happened?",
        "",
        channel="What Would You Do Clips",
    )
    assert row["title_wwyd_candidate_cue"] is True
    assert row["official_wwyd_channel_cue"] is False
    assert row["witnessed_official_wwyd_channel_v1"] is False
    assert row["audited_strict_organic_exclusion_candidate"] is False
