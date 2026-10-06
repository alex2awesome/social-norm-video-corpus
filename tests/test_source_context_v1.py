import json

import pytest

from weaksup.source_context_contract_v1 import packet_prompt, validate_result
from weaksup.build_source_context_packets_v1 import build_packet
from weaksup.run_source_context_llm_v1 import parse_result


def result(**overrides):
    base = {
        "content_type": "organic_capture", "norm_domain": "interpersonal_social",
        "inferred_behavior": "a man cuts an airport queue",
        "inferred_norm": "wait your turn in line", "polarity": "violation",
        "crowd_stance": "condemns_actor", "evidence_fields": ["title", "comments"],
        "note": "framing names the transgression",
    }
    base.update(overrides)
    return base


def test_contract_derives_composites_in_code():
    out = validate_result(result())
    assert out["social_norm_apparent"] == "yes"
    assert out["organic_capture_ok"] is True
    assert out["reroute_hint"] is None
    assert out["acceptance_label"] is None
    staged = validate_result(result(content_type="staged_prank_or_experiment"))
    assert staged["organic_capture_ok"] is False
    assert staged["reroute_hint"] == "instructional_review"
    fight = validate_result(result(norm_domain="violence_crime", inferred_behavior=None,
                                   inferred_norm=None, polarity="none"))
    assert fight["social_norm_apparent"] == "no"


def test_contract_rejects_bad_fields():
    with pytest.raises(ValueError, match="content_type"):
        validate_result(result(content_type="cool_video"))
    with pytest.raises(ValueError, match="evidence_fields"):
        validate_result(result(evidence_fields=["vibes"]))
    with pytest.raises(ValueError, match="inferred_behavior"):
        validate_result(result(inferred_behavior="   "))


def test_parse_result_extracts_json_from_noise():
    text = "Sure! " + json.dumps(result()) + "\nDone."
    assert parse_result(text)["norm_domain"] == "interpersonal_social"
    with pytest.raises(ValueError):
        parse_result("no json here")


def test_packet_and_prompt(tmp_path):
    root = tmp_path
    (root / "data" / "transcripts").mkdir(parents=True)
    (root / "data" / "transcripts" / "reddit__abc.json").write_text(json.dumps(
        {"segments": [{"text": "he just walked past everyone"}]}))
    packet = build_packet(
        "reddit__abc", ["witnessed"],
        {"uid": "reddit__abc", "title": None, "description": "d" * 3000},
        {"post": {"subreddit": "PublicFreakout", "title": "Guy cuts the line",
                  "link_flair_text": "Justified Freakout"},
         "comments": [{"body": "what an entitled jerk", "score": 900}]},
        root, [{"matched_text": "hey, stop"}],
    )
    assert packet["subreddit"] == "PublicFreakout"
    assert packet["title"] == "Guy cuts the line"  # falls back to post title
    assert len(packet["description"]) == 1500
    assert packet["reaction_excerpts"] == ["hey, stop"]
    assert "comments" in packet["context_available"]
    prompt = packet_prompt(packet)
    assert "r/PublicFreakout" in prompt and "entitled jerk" in prompt
    assert "detected reaction" in prompt
