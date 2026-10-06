import pytest

from scripts.select_instructional_retro_duration_holdout_v1 import (
    blind_record, cohort_availability, cohort_name, select,
)


def row(index, source, duration, uid=None):
    return {
        "item_id": f"instructional:{uid or f'u{index}'}:{index}",
        "uid": uid or f"u{index}",
        "query_source": source,
        "duration_hint": duration,
        "source_clip": f"/{index}.mp4",
        "norm": "sealed",
    }


def counts(value=1):
    return {
        "signal_positive": value,
        "short_retro_boundary": value,
        "duration_matched_nonretro_control": value,
    }


def test_frozen_cohort_definition():
    assert cohort_name(row(0, "retro_instr_scan", 5), "retro_instr_scan", 5) == "signal_positive"
    assert cohort_name(row(1, "retro_instr_scan", 4.99), "retro_instr_scan", 5) == "short_retro_boundary"
    assert cohort_name(row(2, "other", 5), "retro_instr_scan", 5) == "duration_matched_nonretro_control"
    assert cohort_name(row(3, "other", 2), "retro_instr_scan", 5) == "ineligible"


def test_selection_is_deterministic_excluded_and_source_disjoint():
    rows = [
        row(0, "retro_instr_scan", 8, "shared"),
        row(1, "retro_instr_scan", 3, "shared"),
        row(2, "retro_instr_scan", 8),
        row(3, "retro_instr_scan", 3),
        row(4, "other", 8),
        row(5, "other", 8),
    ]
    first = select(rows, {"u5"}, counts(), seed="seed")
    second = select(rows, {"u5"}, counts(), seed="seed")
    assert [value["item_id"] for value in first] == [value["item_id"] for value in second]
    assert len({value["uid"] for value in first}) == 3
    assert {value["holdout_cohort"] for value in first} == set(counts())
    assert "u5" not in {value["uid"] for value in first}


def test_selection_allocates_rare_cohorts_before_shared_control_uids():
    rows = [
        row(0, "retro_instr_scan", 3, "short"),
        row(1, "other", 8, "short"),
        row(2, "retro_instr_scan", 8, "long"),
        row(3, "other", 8, "long"),
        row(4, "other", 8, "control"),
    ]
    chosen = select(rows, set(), counts(), seed="rare-first")
    by_cohort = {value["holdout_cohort"]: value["uid"] for value in chosen}
    assert by_cohort == {
        "short_retro_boundary": "short",
        "signal_positive": "long",
        "duration_matched_nonretro_control": "control",
    }


def test_insufficient_cohort_fails_closed():
    with pytest.raises(ValueError, match="insufficient source-disjoint"):
        select([row(0, "retro_instr_scan", 8)], set(), counts(), seed="seed")


def test_blind_record_drops_signal_and_semantics():
    value = blind_record(row(0, "retro_instr_scan", 8), 2)
    assert value["candidate_id"] == "retro-duration-holdout-0002"
    assert "query_source" not in value
    assert "duration_hint" not in value
    assert "norm" not in value


def test_availability_reports_source_exhaustion_without_reusing_uids():
    rows = [
        row(0, "retro_instr_scan", 8, "used-long"),
        row(1, "retro_instr_scan", 4, "used-short"),
        row(2, "retro_instr_scan", 7, "fresh-long"),
        row(3, "other", 9, "fresh-control"),
    ]
    report = cohort_availability(rows, {"used-long", "used-short"})

    assert report["before_exclusion"]["signal_positive"]["source_uids"] == 2
    assert report["after_exclusion"]["signal_positive"]["source_uids"] == 1
    assert report["before_exclusion"]["short_retro_boundary"]["source_uids"] == 1
    assert report["after_exclusion"]["short_retro_boundary"]["source_uids"] == 0
    assert report["after_exclusion"]["duration_matched_nonretro_control"]["source_uids"] == 1
    assert report["policy"] == "diagnostic_only_no_selection_or_corpus_mutation"
