"""Live mode tests: the sliding-window scorer on a fake live stream, plus the small helpers."""

import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from clipper import live
from clipper.config import load_config
from helpers import spec_config
from clipper.signals import ChatMessage

DURATION = 1800
MOMENTS = [700, 1150, 1500]


def fake_stream(seed=1):
    """Background chat (~1.5 msg/s) and quiet audio, with 3 planted moments: the streamer
    gets loud right away and chat explodes ~6 s later (chat lag)."""
    rng = random.Random(seed)
    users = [f"user{i}" for i in range(300)]
    msgs = []
    for sec in range(DURATION):
        for _ in range(rng.choice([0, 1, 1, 2, 2, 3])):
            msgs.append(ChatMessage(sec + rng.random(), rng.choice(users), rng.choice(["hi", "nice", "lol ok"])))
    for t in MOMENTS:
        for sec in range(t + 6, t + 11):
            for _ in range(15):
                msgs.append(ChatMessage(sec + rng.random(), rng.choice(users), rng.choice(["KEKW", "clip it", "?????"])))
    msgs.sort(key=lambda m: m.t)
    nrng = np.random.default_rng(seed)
    audio = -35 + nrng.normal(0, 1.5, DURATION)
    for t in MOMENTS:
        audio[t - 1:t + 3] = -8
    return msgs, audio


def simulate(cfg, msgs, audio, step=5, segment=10):
    """Play the fake stream into LiveScorer like live mode does: chat arrives in real time,
    audio arrives in 10-s pieces once each piece is finished. Returns [(peak, emitted_at)]."""
    scorer = live.LiveScorer(cfg)
    emitted, mi = [], 0
    for now in range(step, DURATION + 1, step):
        new = []
        while mi < len(msgs) and msgs[mi].t <= now:
            new.append(msgs[mi])
            mi += 1
        scorer.add_messages(new)
        seg_start = (now // segment - 1) * segment       # the piece that just finished
        if now % segment == 0 and seg_start >= 0:
            scorer.add_audio(seg_start, audio[seg_start:seg_start + segment])
        recorded = (now // segment) * segment
        for c in scorer.confirmed_peaks(min(now, recorded)):
            emitted.append((c.peak, now, c))
    return emitted


def test_live_scorer_finds_each_moment_once_after_confirm_delay():
    cfg = spec_config()
    msgs, audio = fake_stream()
    emitted = simulate(cfg, msgs, audio)
    peaks = [p for p, _, _ in emitted]
    for t in MOMENTS:
        hits = [(p, at) for p, at, _ in emitted if abs(p - t) <= 3]
        assert len(hits) == 1, f"moment {t}: emitted {hits} (all: {peaks})"
        p, at = hits[0]
        # Never before the reaction has played out.
        assert at >= p + cfg["live"]["confirm_seconds"]
        # ...but not much later either: well inside the ~5 minute goal.
        assert at - p <= 90
    # Never the same moment twice.
    for i, a in enumerate(peaks):
        for b in peaks[i + 1:]:
            assert abs(a - b) >= cfg["peaks"]["min_gap_seconds"]
    # Bounds are absolute stream seconds around the peak.
    for p, _, c in emitted:
        assert c.start < p < c.end


def test_no_peaks_during_warmup():
    cfg = spec_config()
    msgs, audio = fake_stream()
    scorer = live.LiveScorer(cfg)
    scorer.add_messages([m for m in msgs if m.t < 200])
    scorer.add_audio(0, audio[:200])
    assert scorer.confirmed_peaks(200) == []      # < warmup_seconds of history


def test_restart_does_not_repeat_moments():
    """Live mode restarted mid-stream reloads emitted peaks and must not clip them again."""
    cfg = spec_config()
    msgs, audio = fake_stream()
    first = simulate(cfg, msgs, audio)
    scorer = live.LiveScorer(cfg)
    scorer.emitted = [p for p, _, _ in first]
    scorer.add_messages(msgs)
    scorer.add_audio(0, audio)
    assert scorer.confirmed_peaks(DURATION) == []


def test_parse_irc_privmsg():
    assert live.parse_irc_privmsg(":Some_User!some_user@some_user.tmi.twitch.tv PRIVMSG #chan :KEKW no way") \
        == ("some_user", "KEKW no way")
    assert live.parse_irc_privmsg("@badge-info=;color=#FF0000 :a!a@a.tmi.twitch.tv PRIVMSG #c :hi") == ("a", "hi")
    assert live.parse_irc_privmsg(":tmi.twitch.tv 001 justinfan123 :Welcome") is None


def test_read_new_chat_only_complete_lines(tmp_path):
    path = tmp_path / "chat.jsonl"
    path.write_text('{"t": 1.5, "user": "a", "text": "hi"}\n{"t": 2.0, "user": "b", "te', encoding="utf-8")
    msgs, pos = live.read_new_chat(path, 0)
    assert [(m.t, m.user) for m in msgs] == [(1.5, "a")]
    with path.open("a", encoding="utf-8") as f:
        f.write('xt": "yo"}\n')
    msgs, pos2 = live.read_new_chat(path, pos)
    assert [(m.user, m.text) for m in msgs] == [("b", "yo")] and pos2 > pos


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="needs ffmpeg")
def test_buffer_cut_joins_pieces(tmp_path):
    cfg = spec_config()
    cfg["live"]["segment_seconds"] = 10       # this test makes 10-s pieces
    start = datetime.now(timezone.utc).replace(microsecond=0) - timedelta(minutes=5)
    buf = live.StreamBuffer("https://www.twitch.tv/x", tmp_path / "buffer", start, cfg, "x")
    for i in range(3):   # three 10-s pieces named by local wall-clock time, like ffmpeg's strftime
        t = (start + timedelta(seconds=10 * i)).astimezone()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30",
                        "-f", "lavfi", "-i", "sine=frequency=440", "-t", "10", "-c:v", "libx264",
                        "-preset", "ultrafast", "-c:a", "aac", "-f", "mpegts",
                        str(buf.folder / f"seg_{t:%Y%m%d_%H%M%S}.ts")], check=True)
    assert [round(s) for s, _ in buf.segments()] == [0, 10, 20]
    assert buf.recorded_until() == 30          # ffmpeg isn't running, so all pieces are finished
    source_start = buf.cut(12, 25, tmp_path / "cut.mp4")
    assert source_start == 10                  # starts at the piece that contains second 12
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                          str(tmp_path / "cut.mp4")], capture_output=True, text=True, check=True)
    assert abs(float(out.stdout) - 20) < 0.5   # pieces 10-20 and 20-30
    buf.delete_older_than(40)
    assert [round(s) for s, _ in buf.segments()] == [20]   # keeps a little margin


@pytest.mark.skipif(sys.platform != "win32", reason="Windows job objects")
def test_child_dies_with_parent():
    """If Python is killed, the ffmpeg recorder must not keep running (and filling the disk)."""
    code = ("import subprocess, sys, time; from clipper.live import tie_to_this_process; "
            "p = subprocess.Popen(['ping', '-n', '60', '127.0.0.1'], stdout=subprocess.DEVNULL); "
            "tie_to_this_process(p); print(p.pid, flush=True); time.sleep(60)")
    parent = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                              cwd=str(Path(live.__file__).parent.parent))
    child = int(parent.stdout.readline())
    parent.kill()                                          # like closing the terminal / a crash
    parent.wait()
    for _ in range(50):
        alive = str(child) in subprocess.run(["tasklist", "/FI", f"PID eq {child}", "/NH"],
                                             capture_output=True, text=True).stdout
        if not alive:
            break
        time.sleep(0.1)
    assert not alive

def test_live_viewer_clips_make_a_quiet_moment_count():
    """A moment with ordinary chat but several viewer clips must still be confirmed."""
    cfg = spec_config()
    msgs, audio = fake_stream()
    scorer = live.LiveScorer(cfg)
    scorer.add_messages([m for m in msgs])
    scorer.add_audio(0, audio)
    quiet = 1300                                  # nothing special in chat or audio here
    scorer.add_clips([{"id": f"c{i}", "vod_offset": quiet - 18 + i, "duration": 30, "views": 5} for i in range(6)])
    peaks = [c.peak for c in scorer.confirmed_peaks(1400)]   # window 500-1400 s
    assert any(abs(p - quiet) <= 15 for p in peaks)   # the clip bump is ~8 s wide


def test_live_clip_times_from_creation_time(monkeypatch):
    start = datetime(2026, 11, 19, 18, 0, tzinfo=timezone.utc)
    checker = live.LiveChecker()
    checker.twitch_id, checker.twitch_secret, checker.token = "id", "secret", "tok"
    checker._user_ids = {"speed": "42"}
    created = (start + timedelta(seconds=630)).isoformat().replace("+00:00", "Z")
    monkeypatch.setattr(live.fetch, "_helix_get", lambda path, params, cid, tok: {"data": [
        {"id": "a", "created_at": created, "duration": 30.0, "view_count": 12}]})
    clips = checker.twitch_live_clips("speed", start, start)
    assert clips == [{"id": "a", "vod_offset": 600.0, "duration": 30.0, "views": 12}]
