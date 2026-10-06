from types import SimpleNamespace

from src import sources


def test_arctic_shift_host_uses_one_proxy_attempt_then_direct(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs.get("proxies"))
        raise RuntimeError("endpoint unavailable")

    monkeypatch.setattr(sources.scrape, "_load_proxy_pool", lambda cfg: ["http://one", "http://two"])
    monkeypatch.setattr(sources.requests, "get", fake_get)
    cfg = {
        "api": {
            "use_proxy": True,
            "host_attempt_limits": {"arctic-shift.photon-reddit.com": 1},
        },
        "network": {"proxy_attempts": 6},
    }
    assert sources._get(
        "https://arctic-shift.photon-reddit.com/api/posts/search", {}, cfg, timeout=1
    ) is None
    assert len(calls) == 2
    assert calls[0] is not None
    assert calls[1] is None


def test_other_hosts_retain_default_attempt_count(monkeypatch):
    calls = []

    def fake_get(url, **kwargs):
        calls.append(kwargs.get("proxies"))
        return SimpleNamespace(status_code=503)

    monkeypatch.setattr(sources.scrape, "_load_proxy_pool", lambda cfg: ["http://one", "http://two"])
    monkeypatch.setattr(sources.requests, "get", fake_get)
    cfg = {
        "api": {"use_proxy": True, "host_attempt_limits": {}},
        "network": {"proxy_attempts": 2},
    }
    assert sources._get("https://api.example.test/search", {}, cfg, timeout=1) is None
    assert len(calls) == 3
