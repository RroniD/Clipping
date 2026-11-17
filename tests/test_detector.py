"""End-to-end detector test on synthetic data (CLAUDE.md "Testing").

We fake a 1-hour stream: steady background chat and quiet audio, plus three planted
moments. At each moment the streamer gets loud right away, and chat explodes a few
seconds later (chat lag). The detector must find every moment within +/- 3 s.
"""

import random
import shutil
import wave

import numpy as np
import pytest

from clipper.config import load_config
from clipper.scoring import find_candidates
from clipper.signals import ChatMessage, audio_loudness, chat_signals

DURATION = 3600
MOMENTS = [600, 1500, 2700]
SAMPLE_RATE = 8000


def fake_chat(rng: random.Random) -> list[ChatMessage]:
    users = [f"user{i}" for i in range(300)]
    boring = ["hello", "what game is this", "nice car", "how long has he been live", "lol ok"]
    msgs = []
    for sec in range(DURATION):
        for _ in range(rng.choice([0, 1, 1, 2, 2, 3])):  # ~1.5 msg/s background
            msgs.append(ChatMessage(sec + rng.random(), rng.choice(users), rng.choice(boring)))
    for t in MOMENTS:
        # Reaction starts ~6 s after the moment (default chat lag) and lasts ~5 s.
        for sec in range(t + 6, t + 11):
            for _ in range(15):
                msgs.append(ChatMessage(sec + rng.random(), rng.choice(users),
                                        rng.choice(["KEKW", "OMEGALUL KEKW", "clip it", "?????", "💀💀"])))
    msgs.sort(key=lambda m: m.t)
    return msgs


def fake_audio(path, rng: np.random.Generator) -> None:
    """Quiet noise (~-35 dB) with 4-second loud bursts (~-8 dB) starting at each moment."""
    samples = rng.normal(0, 0.02, DURATION * SAMPLE_RATE)
    for t in MOMENTS:
        a, b = (t - 1) * SAMPLE_RATE, (t + 3) * SAMPLE_RATE
        samples[a:b] = rng.normal(0, 0.4, b - a)
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def assert_found(candidates, tolerance=3):
    peaks = [c.peak for c in candidates]
    for t in MOMENTS:
        assert any(abs(p - t) <= tolerance for p in peaks), f"moment at {t}s not found, peaks={peaks}"
    # The planted moments should be the top 3, not buried under noise.
    for t in MOMENTS:
        assert any(abs(c.peak - t) <= tolerance for c in candidates[:3])


def test_chat_only_finds_moments_with_lag_corrected():
    cfg = load_config(None)
    raw = chat_signals(fake_chat(random.Random(1)), DURATION, cfg)
    _, _, candidates = find_candidates(raw, cfg)
    assert_found(candidates)


def test_chat_lag_matters():
    """Without lag correction the peaks land ~6 s late, outside the +/- 3 s tolerance."""
    cfg = load_config(None)
    cfg["chat"]["lag_seconds"] = 0
    raw = chat_signals(fake_chat(random.Random(1)), DURATION, cfg)
    _, _, candidates = find_candidates(raw, cfg)
    top = sorted(c.peak for c in candidates[:3])
    assert all(p - t > 3 for p, t in zip(top, MOMENTS))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_audio_only_finds_moments(tmp_path):
    cfg = load_config(None)
    wav = tmp_path / "audio.wav"
    fake_audio(wav, np.random.default_rng(2))
    loud = audio_loudness(wav, SAMPLE_RATE)
    assert abs(len(loud) - DURATION) <= 1
    _, _, candidates = find_candidates({"audio": loud[:DURATION]}, cfg)
    assert_found(candidates)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_chat_and_audio_together(tmp_path):
    cfg = load_config(None)
    wav = tmp_path / "audio.wav"
    fake_audio(wav, np.random.default_rng(3))
    raw = chat_signals(fake_chat(random.Random(3)), DURATION, cfg)
    raw["audio"] = audio_loudness(wav, SAMPLE_RATE)[:DURATION]
    _, score, candidates = find_candidates(raw, cfg)
    assert_found(candidates)
    # Every candidate's rough bounds contain its peak and stay inside the VOD.
    for c in candidates:
        assert 0 <= c.start < c.peak < c.end <= DURATION
