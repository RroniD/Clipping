"""Phone notifications: sent as JSON to ntfy.sh, never crash clipping, topic kept in .env."""

import requests

from clipper import notify


class FakeResponse:
    ok = True


def test_clip_ready_message(monkeypatch):
    sent = []
    monkeypatch.setenv("NTFY_TOPIC", "autoclipper-test")
    monkeypatch.setattr(notify.requests, "post", lambda url, json, timeout: (sent.append(json), FakeResponse())[1])
    d = {"peak_time": "1h02m03s", "warning": False, "ai": {"score": 8, "hook": "He got 5 stars 🚔"}}
    assert notify.clip_ready("speed", d, "http://192.168.0.5:8000")
    msg = sent[0]
    assert msg["topic"] == "autoclipper-test" and msg["click"] == "http://192.168.0.5:8000"
    assert "AI 8/10" in msg["message"] and "🚔" in msg["message"] and msg["priority"] == 4


def test_warning_clips_say_so(monkeypatch):
    sent = []
    monkeypatch.setenv("NTFY_TOPIC", "t")
    monkeypatch.setattr(notify.requests, "post", lambda url, json, timeout: (sent.append(json), FakeResponse())[1])
    notify.clip_ready("speed", {"warning": True, "ai": None}, None)
    assert "SPOILERS" in sent[0]["message"] and sent[0]["tags"] == ["warning"]


def test_no_topic_or_no_internet_never_raises(monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    assert notify.send("t", "m") is False
    monkeypatch.setenv("NTFY_TOPIC", "t")

    def boom(*a, **k):
        raise requests.ConnectionError("offline")
    monkeypatch.setattr(notify.requests, "post", boom)
    assert notify.send("t", "m") is False


def test_topic_is_created_once_in_env(tmp_path, monkeypatch):
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    env = tmp_path / ".env"
    env.write_text("TWITCH_CLIENT_ID=x\n", encoding="utf-8")
    first = notify.ensure_topic(env)
    monkeypatch.delenv("NTFY_TOPIC", raising=False)
    assert notify.ensure_topic(env) == first and first.startswith("autoclipper-") and len(first) > 25
    assert env.read_text(encoding="utf-8").count("NTFY_TOPIC=") == 1
