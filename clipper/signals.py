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
        # The login name (not the numeric id), so bots like "fossabot" can be recognised and
        # so it matches what the live recorders save.
        return [ChatMessage(t=float(c["content_offset_seconds"]),
                            user=str((c.get("commenter") or {}).get("name")
                                     or (c.get("commenter") or {}).get("_id", "")).lower(),
                            text=(c.get("message") or {}).get("body", "") or "")
                for c in data["comments"]]
    # Kick recorder format: [{"t": ..., "user": ..., "text": ...}]
    return [ChatMessage(t=float(m["t"]), user=str(m.get("user", "")), text=m.get("text", "") or "")
            for m in data]


# --------------------------------------------------------------------------- cleaning chat

# Twitch's own notices that end up in chat replays ("X subscribed at Tier 1", gift subs, raids).
_NOTICE = re.compile(r"^\S+ (subscribed (at tier|with prime)|is gifting|gifted (a|\d+) tier|"
                     r"is paying forward|converted from|is continuing the gift|is raiding|raided)", re.I)


def clean_chat(messages: list[ChatMessage], cfg: dict) -> tuple[list[ChatMessage], dict]:
    """Drop chat that says nothing about the moment. Returns (kept messages, what was dropped).

    - bots (Fossabot, Nightbot, ...): they post timers and links on a schedule
    - !commands: "!followage", "!gamble all" are people talking to a bot
    - sub / gift / raid notices: a sub train looks like a chat explosion but isn't a moment
    Spammers are handled separately (a per-user cap in chat_signals), because their
    messages ARE reactions, just too many of them.
    """
    bots = {b.lower() for b in cfg["chat"]["bots"]}
    kept, dropped = [], {"bots": 0, "commands": 0, "notices": 0}
    for m in messages:
        text = m.text.strip()
        if m.user.lower() in bots:
            dropped["bots"] += 1
        elif text.startswith("!"):
            dropped["commands"] += 1
        elif _NOTICE.match(text):
            dropped["notices"] += 1
        else:
            kept.append(m)
    return kept, dropped


# --------------------------------------------------------------------------- chat mood

def _tokens(text: str) -> list[str]:
    return [t for t in (_PUNCT.sub("", tok) for tok in text.lower().split()) if t]


def moods_of(text: str, moods: dict[str, list[str]]) -> set[str]:
    """Which moods a message shows: funny, hype, tense, confusion, fail (can be several)."""
    tokens = _tokens(text)
    squashed = {re.sub(r"(.)\1+$", r"\1", t) for t in tokens}   # "KEKWWW" -> "kekw"
    lowered = text.lower()
    found = set()
    for mood, words in moods.items():
        for w in words:
            w = w.lower()
            if (" " in w and w in lowered) or w in tokens or w in squashed:
                found.add(mood)
                break
    if re.search(r"\?{2,}", text):
        found.add("confusion")
    return found


def mood_mix(messages: list[ChatMessage], start: float, end: float, cfg: dict) -> dict[str, int]:
    """Share (%) of each mood among mood-carrying messages in [start, end + chat lag].
    The judge gets this, e.g. {"funny": 70, "confusion": 20, "hype": 10}: a funny moment and a
    shocking one look the same in message counts, but not in what people type."""
    lag = cfg["chat"]["lag_seconds"]
    counts: dict[str, int] = {}
    for m in messages:
        if start <= m.t - lag <= end:
            for mood in moods_of(m.text, cfg["chat"]["moods"]):
                counts[mood] = counts.get(mood, 0) + 1
    total = sum(counts.values())
    return {k: round(100 * v / total) for k, v in sorted(counts.items(), key=lambda kv: -kv[1])} if total else {}


def _norm(text: str) -> str:
    """Normalise a message for spam-wave matching: lowercase, squash repeats, first 4 words."""
    toks = [re.sub(r"(.)\1{2,}", r"\1\1", t) for t in _tokens(text)]
    return " ".join(toks[:4])


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
    spam = np.zeros(n_seconds)
    user_seconds: set[tuple[int, str]] = set()  # which users spoke in which second
    per_user: dict[tuple[str, int], int] = {}   # (user, 5-s bucket) -> messages counted
    recent: dict[str, list[tuple[float, str]]] = {}   # normalised text -> (time, user) seen
    cap = int(c["per_user_cap"])
    # New chatters: someone typing for the first time (in this stretch of chat). Lurkers only
    # speak up when something big happens. The first 2 minutes are skipped: there, everyone is
    # "new" because we just started reading.
    newcomers = np.zeros(n_seconds)
    caps = np.zeros(n_seconds)      # ALL-CAPS messages: chat shouting
    seen_users: set[str] = set()
    first_t = messages[0].t if messages else 0.0

    for m in messages:
        sec = int(math.floor(m.t - lag))
        if m.user not in seen_users:
            seen_users.add(m.user)
            if m.t - first_t > 120 and 0 <= sec < n_seconds:
                newcomers[sec] += 1
        if not 0 <= sec < n_seconds:
            continue
        user_seconds.add((sec, m.user))
        # One person typing 10 messages in 5 s is one excited person, not ten: count at most
        # `per_user_cap` of their messages per window. (The top chatters in a real Speed VOD
        # sent ~500 messages each.)
        key = (m.user, sec // width)
        per_user[key] = per_user.get(key, 0) + 1
        if per_user[key] > cap:
            continue
        counts[sec] += 1
        if is_hype(m.text, hype_words, hype_phrases, c["hype_ratio"]):
            hype[sec] += 1
        letters = [ch for ch in m.text if ch.isalpha()]
        if len(letters) >= 4 and sum(ch.isupper() for ch in letters) >= 0.8 * len(letters):
            caps[sec] += 1
        # Spam wave: many DIFFERENT people typing the same thing within 10 s ("LMAOOO",
        # "HE DIDNT", a copypasta) is a strong sign everyone saw the same moment.
        norm = _norm(m.text)
        if norm:
            seen = [(t, u) for t, u in recent.get(norm, []) if m.t - t <= 10]
            if any(u != m.user for _, u in seen):
                spam[sec] += 1
            recent[norm] = seen + [(m.t, m.user)]

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

    return {"chat": _window_sum(counts, width), "unique": unique, "hype": _window_sum(hype, width),
            "spam": _window_sum(spam, width), "newcomers": _window_sum(newcomers, width),
            "caps": _window_sum(caps, width)}


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


# --------------------------------------------------------------------------- calibration from viewer clips

def estimate_chat_lag(messages: list[ChatMessage], clips: list[dict], n_seconds: int, cfg: dict,
                      max_lag: int = 40) -> tuple[int, float] | None:
    """How many seconds this streamer's chat reacts after a moment. Returns (lag, correlation).

    Viewer clips mark where moments are (independently of chat). We slide the chat curve
    0..max_lag seconds earlier and keep the shift where it lines up best with the clip curve.
    Big streams with stream delay or slow chat can lag 10-20 s, not the default 6.
    Needs enough clips to mean anything; returns None otherwise.
    """
    if len(clips) < 15 or not messages:
        return None
    no_lag = {**cfg, "chat": {**cfg["chat"], "lag_seconds": 0}}
    chat = robust_z(chat_signals(messages, n_seconds, no_lag)["chat"], int(cfg["chat"]["baseline_seconds"]),
                    float(cfg["signals"]["floor"]["chat"]))
    density = viewer_clip_signal(clips, n_seconds, cfg)
    density = density - density.mean()
    best = None
    for lag in range(0, max_lag + 1):
        a = np.clip(chat[lag:], 0, None)
        b = density[: n_seconds - lag]
        if a.std() == 0 or b.std() == 0:
            continue
        corr = float(np.corrcoef(a, b)[0, 1])
        if best is None or corr > best[1]:
            best = (lag, corr)
    return best


def viewer_clip_bounds(clips: list[dict], peak: float, radius: float = 60, min_clips: int = 2
                       ) -> tuple[float, float] | None:
    """Where viewers themselves cut this moment: the median start and end of their clips that
    lie within +/- radius s of the peak. Many viewers agreeing on a cut is a good first guess."""
    near = [c for c in clips if abs(c["vod_offset"] + c["duration"] / 2 - peak) <= radius]
    if len(near) < min_clips:
        return None
    starts = sorted(c["vod_offset"] for c in near)
    ends = sorted(c["vod_offset"] + c["duration"] for c in near)
    return float(np.median(starts)), float(np.median(ends))
