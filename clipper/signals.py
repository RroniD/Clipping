"""Turn raw inputs (chat, audio, viewer clips) into per-second signals, then z-scores.

Every signal is a numpy array where index i means "second i of the VOD". Using one
shared time axis makes it easy to add the signals together in scoring.py.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from clipper.fetch import require_tool


@dataclass
class ChatMessage:
    t: float      # seconds since the start of the VOD
    user: str
    text: str


# --------------------------------------------------------------------------- loading chat

def load_chat(path: Path) -> list[ChatMessage]:
    """Load chat from either TwitchDownloaderCLI's JSON or our Kick recorder's JSON/JSONL."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        data = json.loads(text)

    if isinstance(data, dict) and "comments" in data:  # TwitchDownloaderCLI format
        return [ChatMessage(t=float(c["content_offset_seconds"]),
                            user=str((c.get("commenter") or {}).get("_id", "")),
                            text=(c.get("message") or {}).get("body", "") or "")
                for c in data["comments"]]
    # Kick recorder format: [{"t": ..., "user": ..., "text": ...}]
    return [ChatMessage(t=float(m["t"]), user=str(m.get("user", "")), text=m.get("text", "") or "")
            for m in data]


# --------------------------------------------------------------------------- hype detection

_PUNCT = re.compile(r"^[^\w?💀😂🤣😭🔥]+|[^\w?💀😂🤣😭🔥]+$")


def is_hype(text: str, hype_words: set[str], hype_phrases: list[str], ratio: float) -> bool:
    """True if a message is mostly hype words/emotes, contains a hype phrase, or is '?????'."""
    lowered = text.lower().strip()
    if not lowered:
        return False
    if any(phrase in lowered for phrase in hype_phrases):
        return True
    tokens = [_PUNCT.sub("", tok) for tok in lowered.split()]
    tokens = [t for t in tokens if t]
    if not tokens:
        return False

    def token_is_hype(tok: str) -> bool:
        if tok in hype_words:
            return True
        if set(tok) == {"?"} and len(tok) >= 2:   # "??", "?????"
            return True
        if re.fullmatch(r"w+", tok):              # "W", "WWWW"
            return True
        # "KEKWWW", "LULLL": strip repeated last letters and try again
        squashed = re.sub(r"(.)\1+$", r"\1", tok)
        return squashed in hype_words

    hype_count = sum(token_is_hype(t) for t in tokens)
    return hype_count / len(tokens) >= ratio


# --------------------------------------------------------------------------- chat signals

def _window_sum(per_second: np.ndarray, width: int) -> np.ndarray:
    """Sum over a centred window of `width` seconds (e.g. 5 s = 2 before, this one, 2 after)."""
    kernel = np.ones(width)
    return np.convolve(per_second, kernel, mode="same")


def chat_signals(messages: list[ChatMessage], n_seconds: int, cfg: dict) -> dict[str, np.ndarray]:
    """Build the `chat`, `unique` and `hype` signals.

    Chat lag: people type *after* something happens, so we move every message
    `lag_seconds` earlier before counting. That lines the chat spike up with the moment.
    """
    c = cfg["chat"]
    lag, width = float(c["lag_seconds"]), int(c["smooth_seconds"])
    hype_words = {w.lower() for w in c["hype_words"]}
    hype_phrases = [p.lower() for p in c["hype_phrases"]]

    counts = np.zeros(n_seconds)
    hype = np.zeros(n_seconds)
    user_seconds: set[tuple[int, str]] = set()  # which users spoke in which second

    for m in messages:
        sec = int(math.floor(m.t - lag))
        if not 0 <= sec < n_seconds:
            continue
        counts[sec] += 1
        user_seconds.add((sec, m.user))
        if is_hype(m.text, hype_words, hype_phrases, c["hype_ratio"]):
            hype[sec] += 1

    # Unique chatters per window: a user counts once for every window that contains
    # at least one of their messages. Collect (window_centre, user) pairs in a set so
    # the same user spamming 10 times in a window still counts once.
    half = width // 2
    covered: set[tuple[int, str]] = set()
    for sec, user in user_seconds:
        for centre in range(sec - half, sec + half + 1):
            if 0 <= centre < n_seconds:
                covered.add((centre, user))
    unique = np.zeros(n_seconds)
    for centre, _ in covered:
        unique[centre] += 1

    return {"chat": _window_sum(counts, width), "unique": unique, "hype": _window_sum(hype, width)}


# --------------------------------------------------------------------------- audio

def audio_loudness(audio_path: Path, sample_rate: int = 8000) -> np.ndarray:
    """Loudness (dB) for each second of audio.

    ffmpeg decodes the file to raw 16-bit mono samples and streams them to us through a
    pipe. We read one second at a time, so a 6-hour VOD never has to fit in memory.
    """
    ffmpeg = require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    cmd = [ffmpeg, "-v", "error", "-i", str(audio_path), "-vn", "-ac", "1",
           "-ar", str(sample_rate), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    bytes_per_second = sample_rate * 2  # 2 bytes per 16-bit sample
    levels = []
    assert proc.stdout is not None
    while True:
        chunk = proc.stdout.read(bytes_per_second)
        if len(chunk) < 2:
            break
        samples = np.frombuffer(chunk[: len(chunk) // 2 * 2], dtype=np.int16).astype(np.float64)
        rms = np.sqrt(np.mean(samples ** 2)) / 32768.0
        # log of 0 is -infinity, so silence is floored at -90 dB.
        levels.append(20 * math.log10(max(rms, 10 ** (-90 / 20))))
    proc.wait()
    if proc.returncode not in (0, None):
        raise RuntimeError(f"ffmpeg could not decode {audio_path}")
    return np.array(levels)


# --------------------------------------------------------------------------- viewer clips

def viewer_clip_signal(clips: list[dict], n_seconds: int, cfg: dict) -> np.ndarray:
    """One Gaussian bump per viewer clip, bigger for clips with more views.

    Viewers press "clip" after the moment and Twitch clips include ~the last 30 s, so the
    moment is usually in the later part of the clip: we centre the bump at 60% of it.
    log10 on the views stops one viral clip from drowning out everything else.
    """
    s = cfg["signals"]
    sigma, centre_frac = float(s["viewer_clip_sigma"]), float(s["viewer_clip_center"])
    t = np.arange(n_seconds)
    signal = np.zeros(n_seconds)
    for c in clips:
        centre = c["vod_offset"] + centre_frac * c["duration"]
        weight = 1 + math.log10(1 + c.get("views", 0))
        lo, hi = max(0, int(centre - 4 * sigma)), min(n_seconds, int(centre + 4 * sigma) + 1)
        signal[lo:hi] += weight * np.exp(-0.5 * ((t[lo:hi] - centre) / sigma) ** 2)
    return signal


# --------------------------------------------------------------------------- robust z-score

def rolling_median_mad(x: np.ndarray, window: int, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    """Median and MAD (median absolute deviation) over a centred window, for every second.

    Why median/MAD instead of mean/std? A big spike pulls the mean and std up, which would
    hide the very spike we're looking for. The median barely moves.
    The ends are padded by repeating the edge value so every second gets a full window.
    Work is done in chunks to keep memory small on long VODs.
    """
    window = max(3, min(window, len(x)))
    half = window // 2
    padded = np.pad(x, (half, half), mode="edge")
    med = np.empty(len(x))
    mad = np.empty(len(x))
    for start in range(0, len(x), chunk):
        stop = min(start + chunk, len(x))
        rows = np.lib.stride_tricks.sliding_window_view(padded[start: stop + 2 * half], 2 * half + 1)
        m = np.median(rows, axis=1)
        med[start:stop] = m
        mad[start:stop] = np.median(np.abs(rows - m[:, None]), axis=1)
    return med, mad


def robust_z(x: np.ndarray, window: int, floor: float) -> np.ndarray:
    """(x - rolling_median) / (1.4826 * rolling_MAD + floor).

    1.4826 makes MAD comparable to a standard deviation for normal data. The floor stops
    a nearly dead chat (MAD = 0) from turning one extra message into a huge z-score.
    """
    if len(x) == 0:
        return x.copy()
    med, mad = rolling_median_mad(x, window)
    return (x - med) / (1.4826 * mad + floor)


def all_z_scores(raw: dict[str, np.ndarray], cfg: dict) -> dict[str, np.ndarray]:
    """Z-score every raw signal with its own floor and baseline window."""
    floors = cfg["signals"]["floor"]
    z = {}
    for name, x in raw.items():
        window = cfg["audio"]["baseline_seconds"] if name == "audio" else cfg["chat"]["baseline_seconds"]
        z[name] = robust_z(x, int(window), float(floors.get(name, 1.0)))
    return z
