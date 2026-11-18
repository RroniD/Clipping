"""Editing streamers.toml / campaigns.toml from the page: careful, valid, keeps your comments."""

import pytest

from clipper import settings_files
from clipper.config import load_campaigns, load_streamers


def test_add_and_remove_streamer_keeps_the_rest(tmp_path):
    path = tmp_path / "streamers.toml"
    path.write_text("# my comment\n[streamers.ishowspeed]\nplatform = \"twitch\"\nfacecam = [0, 398, 457, 256]\n")
    settings_files.add_streamer("kaicenat", "twitch", path)
    assert set(load_streamers(path)) == {"ishowspeed", "kaicenat"}
    with pytest.raises(ValueError):
        settings_files.add_streamer("kaicenat", "twitch", path)           # already there
    with pytest.raises(ValueError):
        settings_files.add_streamer("bad name!", "twitch", path)
    settings_files.remove_streamer("kaicenat", path)
    s = load_streamers(path)
    assert set(s) == {"ishowspeed"} and s["ishowspeed"]["facecam"] == [0, 398, 457, 256]
    assert "# my comment" in path.read_text()


def test_add_campaign_writes_valid_toml(tmp_path):
    path = tmp_path / "campaigns.toml"
    path.write_text("# example below\n")
    key = settings_files.add_campaign({"name": "Kai GTA Clips", "streamer": "kaicenat",
                                       "required_tags": "#kaicenat, @kaicenat", "disclosure": '#ad "quoted"',
                                       "max_seconds": "60", "rate_per_1000": "1.5", "tiktok": True,
                                       "youtube": True}, path)
    c = load_campaigns(path)[key]
    assert key == "kai_gta_clips" and c["streamer"] == "kaicenat"
    assert c["required_tags"] == ["#kaicenat", "@kaicenat"] and c["platforms"] == ["tiktok", "youtube"]
    assert c["disclosure"] == '#ad "quoted"' and c["max_seconds"] == 60 and c["rate_per_1000"] == 1.5
    with pytest.raises(ValueError):
        settings_files.add_campaign({"name": "x", "max_seconds": "a minute"}, path)