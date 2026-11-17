import json

import pytest

from clipper.config import load_config, load_streamers
from clipper.fetch import _parse_twitch_duration, parse_vod_url
from clipper.kick_chat import finalize, message_from_event, parse_time
from clipper.signals import load_chat


def test_parse_vod_url():
    assert parse_vod_url("https://www.twitch.tv/videos/2245678901") == ("twitch", "2245678901")
    assert parse_vod_url("https://www.twitch.tv/somebody/video/123?t=1h") == ("twitch", "123")
    uuid = "1a2b3c4d-1111-2222-3333-444455556666"
    assert parse_vod_url(f"https://kick.com/someone/videos/{uuid}") == ("kick", uuid)
    assert parse_vod_url(f"https://kick.com/video/{uuid}") == ("kick", uuid)
    with pytest.raises(ValueError):
        parse_vod_url("https://youtube.com/watch?v=abc")


def test_twitch_duration():
    assert _parse_twitch_duration("3h12m5s") == 3 * 3600 + 12 * 60 + 5
    assert _parse_twitch_duration("45s") == 45


def test_load_twitch_chat_format(tmp_path):
    p = tmp_path / "chat.json"
    p.write_text(json.dumps({"comments": [
        {"content_offset_seconds": 12.5, "commenter": {"_id": "42"}, "message": {"body": "KEKW"}}]}))
    [m] = load_chat(p)
    assert (m.t, m.user, m.text) == (12.5, "42", "KEKW")


def test_kick_recorder_roundtrip(tmp_path):
    start = parse_time("2026-11-17 18:00:00")
    event = {"content": "W", "created_at": "2026-11-17T18:01:30+00:00", "sender": {"username": "bob"}}
    msg = message_from_event(event, start, start)
    assert msg["t"] == 90.0 and msg["user"] == "bob"
    (tmp_path / "chat.jsonl").write_text(json.dumps(msg) + "\n", encoding="utf-8")
    [m] = load_chat(finalize(tmp_path))
    assert (m.t, m.user, m.text) == (90.0, "bob", "W")


def test_repo_config_files_load():
    cfg = load_config()                 # the real config.toml
    assert cfg["peaks"]["threshold"] == 2.5
    assert cfg["signals"]["floor"]["chat"] > 0
    assert isinstance(load_streamers(), dict)


def test_parse_time_handles_odd_fractions():
    # Python 3.10 rejects fractions that aren't 3 or 6 digits unless we normalise them.
    assert parse_time("2026-11-17T18:01:30.12Z").microsecond == 120000
    assert parse_time("2026-11-17T18:01:30.123456789+00:00").microsecond == 123456
