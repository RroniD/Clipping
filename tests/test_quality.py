"""Tests for the clip-quality upgrades: chat cleaning, moods, spam waves, chat-lag calibration,
viewer-clip bounds, the GTA 6 launch gate and the local frame check."""

import random
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import pytest

from clipper import framecheck, signals
from clipper.config import load_config
from clipper.signals import ChatMessage

CFG = load_config(None)


def test_clean_chat_drops_bots_commands_and_notices():
    msgs = [ChatMessage(1, "fossabot", "Join the Discord!"),
            ChatMessage(2, "a", "!followage"),
            ChatMessage(3, "b", "benz subscribed at Tier 1. They've subscribed for 6 months"),
            ChatMessage(4, "c", "KEKW he crashed"),
            ChatMessage(5, "d", "x is gifting 5 Tier 1 Subs to the community")]
    kept, dropped = signals.clean_chat(msgs, CFG)
    assert [m.text for m in kept] == ["KEKW he crashed"]
    assert dropped == {"bots": 1, "commands": 1, "notices": 2}


def test_per_user_cap_stops_one_spammer_looking_like_a_moment():
    spam = [ChatMessage(100 + i * 0.1, "spammer", f"msg {i}") for i in range(40)]
    sig = signals.chat_signals(spam, 200, CFG)
    lag = int(CFG["chat"]["lag_seconds"])
    assert sig["chat"].max() <= CFG["chat"]["per_user_cap"] * 2   # capped, not 40


def test_spam_wave_needs_different_people():
    wave = [ChatMessage(50 + i * 0.2, f"u{i}", "HE DIDNT LMAOOO") for i in range(10)]
    same = [ChatMessage(150 + i * 0.2, "solo", "HE DIDNT LMAOOO") for i in range(10)]
    sig = signals.chat_signals(wave + same, 200, CFG)
    lag = int(CFG["chat"]["lag_seconds"])
    assert sig["spam"][50 - lag: 53 - lag].sum() > 0
    assert sig["spam"][150 - lag: 153 - lag].sum() == 0


def test_moods():
    moods = CFG["chat"]["moods"]
    assert signals.moods_of("KEKWWW", moods) == {"funny"}
    assert "confusion" in signals.moods_of("?????", moods)
    assert "hype" in signals.moods_of("CLIP IT", moods)
    lag = CFG["chat"]["lag_seconds"]
    msgs = [ChatMessage(10 + lag, "a", "LMAO"), ChatMessage(11 + lag, "b", "KEKW"),
            ChatMessage(12 + lag, "c", "huh")]
    assert signals.mood_mix(msgs, 0, 20, CFG) == {"funny": 67, "confusion": 33}


def test_chat_lag_is_measured_from_viewer_clips():
    """Moments every ~120 s; chat reacts 15 s later; viewer clips mark the moments."""
    rng = random.Random(4)
    n, true_lag = 3600, 15
    moments = list(range(200, 3400, 120))
    msgs = [ChatMessage(s + rng.random(), f"u{rng.randrange(300)}", "hi")
            for s in range(n) for _ in range(rng.choice([0, 1, 2]))]
    for t in moments:
        msgs += [ChatMessage(t + true_lag + rng.random() * 4, f"u{k}", "KEKW") for k in range(40)]
    clips = [{"vod_offset": t - 18, "duration": 30, "views": 10} for t in moments]  # bump at t
    lag, corr = signals.estimate_chat_lag(sorted(msgs, key=lambda m: m.t), clips, n, CFG)
    assert abs(lag - true_lag) <= 2 and corr > 0.2
    assert signals.estimate_chat_lag(msgs, clips[:5], n, CFG) is None   # too few clips to trust


def test_viewer_clip_bounds_use_the_median():
    clips = [{"vod_offset": 90, "duration": 30}, {"vod_offset": 95, "duration": 30},
             {"vod_offset": 100, "duration": 25}, {"vod_offset": 900, "duration": 30}]
    assert signals.viewer_clip_bounds(clips, 110) == (95.0, 125.0)
    assert signals.viewer_clip_bounds(clips[:1], 110) is None


def test_launch_gate():
    before = datetime(2026, 11, 13, tzinfo=timezone.utc)
    after = datetime(2026, 11, 20, tzinfo=timezone.utc)
    assert "pre-release" in framecheck.launch_gate(["GTA 6 EARLY ACCESS!!"], before, CFG)
    assert "pre-release" in framecheck.launch_gate(["Grand Theft Auto VI"], before, CFG)
    assert framecheck.launch_gate(["GTA 6 day 1"], after, CFG) is None
    assert framecheck.launch_gate(["GTA V RP | NoPixel"], before, CFG) is None   # GTA 5 is fine
    assert "unknown" in framecheck.launch_gate(["gta6"], None, CFG)
    assert framecheck.recorded_at({"timestamp": 1763078400}).year == 2025
    assert framecheck.recorded_at({"upload_date": "20261113"}) == before


def _frames(kind: str, n: int = 10) -> np.ndarray:
    rng = np.random.default_rng(1)
    f = rng.integers(60, 200, size=(n, 180, 320)).astype(np.uint8)
    if kind == "letterbox":
        f[:, :20] = 5
        f[:, -20:] = 5
    if kind == "blank":
        f[:] = 12
    return f


def test_frame_stats_and_reasons():
    assert framecheck.framecheck_reasons(framecheck.frame_stats(_frames("normal"), CFG, None), None, CFG) == []
    lb = framecheck.framecheck_reasons(framecheck.frame_stats(_frames("letterbox"), CFG, None), None, CFG)
    assert lb and "black bars" in lb[0]
    blank = framecheck.framecheck_reasons(framecheck.frame_stats(_frames("blank"), CFG, None), None, CFG)
    assert any("loading" in r for r in blank)


def test_hud_missing_is_relative_to_this_streams_hud():
    box = (0.0, 0.7, 0.2, 0.3)
    frames = _frames("normal", 12)
    frames[3:10, 126:, :64] = 100            # minimap region goes flat for 7 s (cutscene)
    stats = framecheck.frame_stats(frames, CFG, box)
    ref = float(np.percentile(stats["hud"], 75))     # what run.py uses as "HUD visible"
    reasons = framecheck.framecheck_reasons(stats, ref, CFG)
    assert any("HUD" in r for r in reasons)
    assert framecheck.hud_box_fractions({"hud_box": [0, 756, 384, 324]}) == pytest.approx(box)


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_gray_frames_reads_one_per_second(tmp_path):
    video = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=640x360:rate=30",
                    "-t", "6", "-c:v", "libx264", "-preset", "ultrafast", str(video)], check=True)
    frames = framecheck.gray_frames(video, 1, 5)
    assert frames.shape[1:] == (180, 320) and 3 <= len(frames) <= 5
