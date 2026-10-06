from scripts.visual_audit_sample import requested_rows


def test_zero_count_does_not_call_expensive_loader():
    def fail():
        raise AssertionError("loader should not be called")

    assert requested_rows(0, fail) == []


def test_positive_count_calls_loader():
    assert requested_rows(2, lambda: ["a", "b"]) == ["a", "b"]
