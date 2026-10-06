from scripts.export_related_snowball_transfer_audit_v1 import parent_uid, stable_key


def test_parent_uid_and_sampling_key_are_deterministic() -> None:
    assert parent_uid("x123") == "dailymotion__x123"
    assert stable_key("child") == stable_key("child")
    assert stable_key("child") != stable_key("other")
