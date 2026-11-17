"""Stage 4: live mode. Watch the allow-listed streamers and clip moments while they're live.

Usage (PowerShell, from the project folder; leave it running, Ctrl+C to stop):
    python -m clipper.live

What happens while a streamer is live:
  1. chat is recorded (Twitch: anonymous IRC; Kick: clipper.kick_chat's websocket)
  2. the stream is recorded into a ROLLING buffer of 10-second pieces; pieces older than
     a few minutes are deleted, so disk use stays small (a few hundred MB)
  3. every few seconds the same signals + scoring as Stage 1 run on the last 15 minutes
  4. once a peak is CONFIRMED (its reaction has played out, ~45 s later) its section is cut
     from the buffer -> Whisper -> snap to speech -> AI judge -> Stage 2 edit -> review queue
  Goal: a finished clip in your queue (as "pending") within ~5 minutes of the moment.
Nothing is ever posted: clips only go to the review queue (hard rule 1).

Design choices (and why):
- A rolling buffer of the LIVE stream, not the VOD. Twitch's VOD of a running stream can be
  downloaded, but only if the streamer has VODs on (many big streamers turn them off or make
  them sub-only), and Kick has no VOD until the stream ends. Recording the live stream works
  for both platforms. `ffmpeg -c copy` only saves the data, so it costs almost no CPU.
  Downside: Twitch inserts ads into the live stream for viewers without Turbo. If the streamer
  runs an ad break, those seconds in the buffer are the ad. Peaks almost never happen during
  ad breaks (chat goes quiet), but check clips in the queue as always.
- Twitch chat via anonymous IRC (WebSocket wss://irc-ws.chat.twitch.tv, login "justinfanNNNN"),
  not EventSub. Twitch's EventSub chat subscription (channel.chat.message) needs a USER access
  token with the user:read:chat scope, which means an OAuth login flow and storing a refresh
  token. Anonymous IRC is read-only, needs no account or token, and Twitch still supports it
  (checked Nov 2026). If Twitch ever shuts it off, EventSub is the replacement.
- Time axis = seconds since the stream started (from the platform API), the same as a VOD's
  offset. So the saved chat.json works with `run.py <VOD url> --chat ...` later.
- Twitch's "create clip" API (POST /helix/clips) is NOT used: it needs a user OAuth token
  with the clips:edit scope and only captures the last ~30 s around "now". We already have
  the moment in our buffer, so it would add a login flow for no gain.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from clipper import fetch, signals
from clipper.config import PROJECT_ROOT, load_config, load_streamers
from clipper.scoring import Candidate, clamp_length, find_candidates, snap_to_speech
from clipper.signals import ChatMessage

OUT_ROOT = PROJECT_ROOT / "out"
TWITCH_IRC = "wss://irc-ws.chat.twitch.tv:443"
SEGMENT_NAME = re.compile(r"seg_(\d{8}_\d{6})\.ts$")


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def log(streamer: str, text: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {streamer}: {text}", flush=True)


# --------------------------------------------------------------------------- is anyone live?

class LiveChecker:
    """Asks Twitch / Kick which allow-listed streamers are live right now."""

    def __init__(self):
        self.twitch_id = os.getenv("TWITCH_CLIENT_ID")
        self.twitch_secret = os.getenv("TWITCH_CLIENT_SECRET")
        self.token = None

    def twitch(self, logins: list[str]) -> dict[str, dict]:
        """{login: {"start": datetime, "title": str}} for the logins that are live.
        Helix GET /streams takes up to 100 logins per request and only returns live ones."""
        if not logins:
            return {}
        if not (self.twitch_id and self.twitch_secret):
            raise SystemExit("Live mode needs TWITCH_CLIENT_ID and TWITCH_CLIENT_SECRET in .env "
                             "to see who is live on Twitch.")
        for attempt in range(2):
            if self.token is None:
                self.token = fetch.twitch_app_token(self.twitch_id, self.twitch_secret)
            try:
                data = fetch._helix_get("streams", {"user_login": logins, "first": 100},
                                        self.twitch_id, self.token)["data"]
                break
            except Exception as exc:
                # App tokens expire after ~60 days; a 401 means "get a new one" and try again.
                if attempt == 0 and "401" in str(exc):
                    self.token = None
                    continue
                raise
        return {s["user_login"].lower(): {
                    "start": datetime.fromisoformat(s["started_at"].replace("Z", "+00:00")),
                    "title": s.get("title", ""), "game": s.get("game_name", "")}
                for s in data if s.get("type") == "live"}

    @staticmethod
    def kick(slug: str) -> dict | None:
        from clipper.kick_chat import lookup_channel
        info = lookup_channel(slug)
        if info["stream_start"] is None:
            return None
        return {"start": info["stream_start"], "title": "", "chatroom_id": info["chatroom_id"],
                "channel_id": info["channel_id"]}


# --------------------------------------------------------------------------- chat recording

def parse_irc_privmsg(line: str) -> tuple[str, str] | None:
    """':nick!nick@nick.tmi.twitch.tv PRIVMSG #channel :message text' -> (nick, text)."""
    m = re.match(r"^(?:@\S+ )?:(\w+)!\S+ PRIVMSG #\w+ :(.*)$", line)
    return (m.group(1).lower(), m.group(2)) if m else None


class TwitchChat(threading.Thread):
    """Reads a channel's chat anonymously and appends it to chat.jsonl (crash-safe).

    Each line: {"t": seconds since stream start, "user", "text", "ts"} - the same format as
    the Kick recorder, so everything downstream treats both platforms the same way.
    """

    def __init__(self, login: str, stream_start: datetime, jsonl: Path):
        super().__init__(daemon=True)
        self.login, self.stream_start, self.jsonl = login, stream_start, jsonl
        self.stop = threading.Event()

    def run(self) -> None:
        import websocket
        backoff = 1
        with self.jsonl.open("a", encoding="utf-8") as out:
            while not self.stop.is_set():
                try:
                    ws = websocket.create_connection(TWITCH_IRC, timeout=30)
                    ws.send("PASS SCHMOOPIIE")   # any password works for anonymous logins
                    ws.send(f"NICK justinfan{random.randint(10000, 99999)}")
                    ws.send(f"JOIN #{self.login}")
                    backoff = 1
                    while not self.stop.is_set():
                        try:
                            raw = ws.recv()
                        except websocket.WebSocketTimeoutException:
                            ws.send("PING :tmi.twitch.tv")   # quiet chat: keep the connection alive
                            continue
                        if not raw:
                            break
                        for line in raw.split("\r\n"):
                            if line.startswith("PING"):
                                ws.send("PONG" + line[4:])   # Twitch disconnects us if we don't answer
                            elif line.startswith(":tmi.twitch.tv RECONNECT"):
                                raise ConnectionError("Twitch asked us to reconnect")
                            elif (msg := parse_irc_privmsg(line)) is not None:
                                now = datetime.now(timezone.utc)
                                out.write(json.dumps({"t": round((now - self.stream_start).total_seconds(), 2),
                                                      "user": msg[0], "text": msg[1], "ts": now.isoformat()},
                                                     ensure_ascii=False) + "\n")
                                out.flush()
                    ws.close()
                except Exception as exc:
                    if self.stop.is_set():
                        break
                    log(self.login, f"chat connection problem ({exc}); reconnecting in {backoff}s")
                    self.stop.wait(backoff)
                    backoff = min(backoff * 2, 60)


class KickChat(threading.Thread):
    """Runs clipper.kick_chat's recorder (Stage 1) in the background."""

    def __init__(self, slug: str, live: dict, out_dir: Path):
        super().__init__(daemon=True)
        self.slug, self.live, self.out_dir = slug, live, out_dir
        self.stop = threading.Event()   # the Kick recorder stops by itself when the stream ends

    def run(self) -> None:
        from clipper.kick_chat import record
        try:
            record(self.slug, self.out_dir, self.live["chatroom_id"], self.live["channel_id"],
                   self.live["start"])
        except Exception as exc:
            log(self.slug, f"Kick chat recorder stopped ({exc})")


def read_new_chat(jsonl: Path, position: int) -> tuple[list[ChatMessage], int]:
    """Read the chat lines added since `position` (a byte offset). Returns (messages, new position).
    Only complete lines are read: the recorder may be halfway through writing the last one."""
    if not jsonl.exists():
        return [], position
    with jsonl.open("rb") as f:
        f.seek(position)
        data = f.read()
    end = data.rfind(b"\n") + 1
    msgs = []
    for line in data[:end].decode("utf-8", errors="replace").splitlines():
        if line.strip():
            m = json.loads(line)
            msgs.append(ChatMessage(t=float(m["t"]), user=str(m.get("user", "")), text=m.get("text", "")))
    return msgs, position + end


# --------------------------------------------------------------------------- video buffer

_JOBS: list = []   # keeps the Windows job handles alive for as long as Python runs


def tie_to_this_process(proc: subprocess.Popen) -> None:
    """Make Windows kill `proc` when this Python process ends, even if it crashes.

    Why: the recorder is a separate ffmpeg process. Without this, if Python is killed (or the
    terminal is closed), ffmpeg keeps recording at ~3 GB per hour until the stream ends.
    A Windows "job object" with KILL_ON_JOB_CLOSE ends every process in it as soon as the
    last handle to the job closes, and Windows closes our handle when Python exits.
    """
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BasicLimits), ("IoInfo", ctypes.c_uint64 * 6),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    info = ExtendedLimits()
    info.BasicLimitInformation.LimitFlags = 0x2000          # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = job and k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)) \
        and k32.AssignProcessToJobObject(job, int(proc._handle))   # 9 = ExtendedLimitInformation
    if ok:
        _JOBS.append(job)
    else:
        print(f"  (couldn't tie ffmpeg to this window, error {ctypes.get_last_error()}; "
              "if you close it, stop ffmpeg in Task Manager)")

class StreamBuffer:
    """Records the live stream into 10-second .ts pieces named by wall-clock time.

    Why the time in the file name? ffmpeg writes a piece as soon as its data arrives, so the
    name says when that piece was on screen for viewers - the same clock chat uses.
    """

    def __init__(self, url: str, folder: Path, stream_start: datetime, cfg: dict, name: str):
        self.url, self.folder, self.stream_start, self.cfg, self.name = url, folder, stream_start, cfg["live"], name
        self.proc: subprocess.Popen | None = None
        self.next_try = 0.0
        self.backoff = 5
        folder.mkdir(parents=True, exist_ok=True)

    def ensure_running(self) -> None:
        """Start ffmpeg, or restart it after a disconnect (with a growing wait between tries)."""
        if self.proc is not None and self.proc.poll() is None:
            return
        if time.time() < self.next_try:
            return
        if self.proc is not None:
            log(self.name, f"stream recording stopped (code {self.proc.returncode}); restarting")
        try:
            # yt-dlp turns the channel page into the stream's HLS address. It contains a token
            # that expires, which is why we ask again on every restart.
            hls = fetch.run(fetch.YT_DLP + ["-g", "-f", self.cfg["buffer_format"], "--no-warnings",
                                            self.url], timeout=60).stdout.strip().splitlines()[0]
            ffmpeg = fetch.require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
            self.proc = subprocess.Popen(
                [ffmpeg, "-hide_banner", "-loglevel", "error", "-i", hls, "-map", "0:v:0?", "-map", "0:a:0",
                 "-c", "copy", "-f", "segment", "-segment_time", str(self.cfg["segment_seconds"]),
                 "-reset_timestamps", "1", "-strftime", "1", str(self.folder / "seg_%Y%m%d_%H%M%S.ts")],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            tie_to_this_process(self.proc)
            self.backoff = 5
        except Exception as exc:
            log(self.name, f"can't open the live stream yet ({str(exc)[-200:]}); retrying in {self.backoff}s")
            self.proc = None
            self.next_try = time.time() + self.backoff
            self.backoff = min(self.backoff * 2, 120)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def segment_start(self, path: Path) -> float | None:
        """Stream second at which a piece starts (from its local-time file name)."""
        m = SEGMENT_NAME.search(path.name)
        if not m:
            return None
        local = datetime.strptime(m.group(1), "%Y%m%d_%H%M%S").astimezone()   # names use local time
        return (local - self.stream_start).total_seconds()

    def segments(self) -> list[tuple[float, Path]]:
        """All pieces on disk as (stream second, path), oldest first."""
        found = [(self.segment_start(p), p) for p in self.folder.glob("seg_*.ts")]
        return sorted((s, p) for s, p in found if s is not None)

    def finished_segments(self) -> list[tuple[float, Path]]:
        """Pieces ffmpeg has finished writing: all but the newest one (while recording)."""
        segs = self.segments()
        running = self.proc is not None and self.proc.poll() is None
        return segs[:-1] if running else segs

    def recorded_until(self) -> float:
        """Stream second up to which finished pieces exist."""
        segs = self.finished_segments()
        if not segs:
            return -1.0
        later = self.segments()
        # A finished piece ends where the next one starts.
        idx = len(segs)
        return later[idx][0] if idx < len(later) else segs[-1][0] + self.cfg["segment_seconds"]

    def cut(self, start: float, end: float, out_path: Path) -> float:
        """Join the pieces covering [start, end] into one .mp4. Returns the stream second it starts at.
        -c copy: no re-encoding, so this takes a second or two."""
        segs = self.finished_segments()
        later = self.segments()
        chosen = []
        for i, (s, p) in enumerate(segs):
            seg_end = later[i + 1][0] if i + 1 < len(later) else s + self.cfg["segment_seconds"]
            if seg_end > start and s < end:
                chosen.append((s, p))
        if not chosen:
            raise RuntimeError(f"nothing recorded between {start:.0f}s and {end:.0f}s")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        listing = out_path.with_suffix(".txt")
        # The concat list needs forward slashes and quotes, even on Windows.
        listing.write_text("".join(f"file '{p.resolve().as_posix()}'\n" for _, p in chosen), encoding="utf-8")
        ffmpeg = fetch.require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
        fetch.run([ffmpeg, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
                   "-c", "copy", str(out_path)])
        listing.unlink(missing_ok=True)
        return chosen[0][0]

    def delete_older_than(self, stream_second: float) -> None:
        for s, p in self.finished_segments():
            if s + self.cfg["segment_seconds"] * 2 < stream_second:
                try:
                    p.unlink()
                except OSError:
                    pass   # still open somewhere: try again next time


# --------------------------------------------------------------------------- live scoring

class LiveScorer:
    """Runs Stage 1's signals + scoring on the most recent minutes, again and again.

    It reuses signals.chat_signals and scoring.find_candidates unchanged; the only live-specific
    parts are: score just a sliding window, wait for each peak's reaction to play out before
    trusting it (`confirm_seconds`), and remember emitted peaks so a moment is clipped once.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lv = cfg["live"]
        self.messages: list[ChatMessage] = []
        self.audio = np.full(0, np.nan)          # dB per stream second; NaN = not recorded
        self.first_second: float | None = None   # first second we have any data for
        self.emitted: list[float] = []           # peaks already turned into clips
        self.last_chat_z: tuple[int, np.ndarray] | None = None

    def add_messages(self, msgs: list[ChatMessage]) -> None:
        if msgs:
            self.messages.extend(msgs)
            self._seen(min(m.t for m in msgs))

    def add_audio(self, start_second: float, levels: np.ndarray) -> None:
        a = int(round(start_second))
        if a < 0:
            levels, a = levels[-a:], 0
        need = a + len(levels)
        if need > len(self.audio):
            self.audio = np.concatenate([self.audio, np.full(need - len(self.audio), np.nan)])
        self.audio[a:need] = levels
        finite = np.isfinite(levels)
        if finite.any():
            self._seen(a + int(np.argmax(finite)))   # first second that really has a level

    def _seen(self, t: float) -> None:
        self.first_second = t if self.first_second is None else min(self.first_second, t)

    def confirmed_peaks(self, now: float) -> list[Candidate]:
        """Peaks that are certain at stream second `now`, with absolute rough bounds."""
        lv, clip_cfg = self.lv, self.cfg["clip"]
        if self.first_second is None or now - self.first_second < lv["warmup_seconds"]:
            return []   # not enough history yet to know what "normal" looks like
        w0 = int(max(self.first_second, now - lv["window_seconds"]))
        n = int(now) - w0
        if n <= 0:
            return []
        # Old messages are no longer needed for scoring; keep the list from growing for hours.
        keep_from = w0 - self.cfg["chat"]["lag_seconds"] - 10
        if self.messages and self.messages[0].t < keep_from:
            self.messages = [m for m in self.messages if m.t >= keep_from]

        raw: dict[str, np.ndarray] = {}
        window_msgs = [ChatMessage(m.t - w0, m.user, m.text) for m in self.messages]
        if window_msgs:
            raw.update(signals.chat_signals(window_msgs, n, self.cfg))
        audio = self.audio[w0:w0 + n]
        if len(audio) and np.isfinite(audio).any():
            # Seconds not recorded yet (the newest few) get the typical level, so they look normal.
            filled = np.full(n, np.nanmedian(audio))
            filled[: len(audio)] = np.where(np.isfinite(audio), audio, np.nanmedian(audio))
            raw["audio"] = filled
        if not raw:
            return []
        z, score, cands = find_candidates(raw, self.cfg, top_n=lv["max_peaks_per_check"])
        if "chat" in z:
            self.last_chat_z = (w0, z["chat"])

        out = []
        gap = self.cfg["peaks"]["min_gap_seconds"]
        for c in cands:
            peak = w0 + c.peak
            end = w0 + c.end
            if peak < w0 + lv["edge_seconds"] and w0 > self.first_second:
                continue   # too close to the window's left edge: its baseline is one-sided
            # Confirmed = the reaction has played out AND we've recorded past the clip's end,
            # plus room for snapping the end to the end of a sentence.
            if now < max(peak + lv["confirm_seconds"], end + after_margin(self.cfg)):
                continue
            if any(abs(peak - e) < gap for e in self.emitted):
                continue   # this moment was already clipped
            self.emitted.append(peak)
            out.append(Candidate(peak=peak, peak_score=c.peak_score, start=w0 + c.start, end=end,
                                 rough_start=w0 + c.start, rough_end=end, breakdown=c.breakdown))
        return out

    def chat_z_full(self, length: int) -> np.ndarray | None:
        """The last window's chat z-scores placed on the stream's time axis (for the cutscene check)."""
        if self.last_chat_z is None:
            return None
        w0, z = self.last_chat_z
        full = np.zeros(max(length, w0 + len(z)))
        full[w0:w0 + len(z)] = z
        return full


# --------------------------------------------------------------------------- one live stream

class Session:
    """Everything for one streamer while they're live: chat, buffer, scoring, clip jobs."""

    def __init__(self, name: str, platform: str, live: dict, cfg: dict, jobs: queue.Queue,
                 out_root: Path = OUT_ROOT, save_clips: bool = True):
        self.name, self.platform, self.cfg, self.jobs = name, platform, cfg, jobs
        self.start = live["start"]
        self.title = live.get("title", "")
        self.save_clips = save_clips
        # Hard rule 2: GTA 6 footage from before the launch is pre-release footage.
        from clipper.framecheck import launch_gate
        self.launch_warning = launch_gate([self.title, live.get("game", "")], self.start, cfg)
        if self.launch_warning:
            log(name, f"WARNING: {self.launch_warning}. Every clip from this stream gets the warning flag.")
        self.vod_id = f"live_{self.start.strftime('%Y%m%d_%H%M%S')}"
        self.dir = out_root / name / self.vod_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.url = f"https://www.twitch.tv/{name}" if platform == "twitch" else f"https://kick.com/{name}"
        self.jsonl = self.dir / "chat.jsonl"
        self.chat_pos = 0
        self.scorer = LiveScorer(cfg)
        self.buffer = StreamBuffer(self.url, self.dir / "buffer", self.start, cfg, name)
        self.done_segments: set[str] = set()
        self.offline_checks = 0
        self.last_tick = 0.0
        self._load_previous()
        self.chat = (TwitchChat(name, self.start, self.jsonl) if platform == "twitch"
                     else KickChat(name, live, self.dir))
        self.chat.start()
        (self.dir / "session.json").write_text(json.dumps(
            {"streamer": name, "platform": platform, "url": self.url, "title": self.title,
             "stream_start": self.start.isoformat()}, indent=2), encoding="utf-8")
        log(name, f"is live ('{self.title}'). Recording into {self.dir}")

    def _load_previous(self) -> None:
        """If live mode was restarted during the same stream, pick up where it left off:
        reload the chat and audio so far and don't clip the same moments again."""
        msgs, self.chat_pos = read_new_chat(self.jsonl, 0)
        self.scorer.add_messages(signals.clean_chat(msgs, self.cfg)[0])
        audio_path = self.dir / "audio_db.npy"
        if audio_path.exists():
            self.scorer.add_audio(0, np.load(audio_path))
        cands = self.dir / "candidates.json"
        if cands.exists():
            self.scorer.emitted = [float(d["peak"]) for d in json.loads(cands.read_text(encoding="utf-8"))["candidates"]]

    def now(self) -> float:
        return (datetime.now(timezone.utc) - self.start).total_seconds()

    def tick(self) -> None:
        """Called every couple of seconds by the main loop."""
        self.buffer.ensure_running()
        if time.time() - self.last_tick < self.cfg["live"]["check_seconds"]:
            return
        self.last_tick = time.time()

        # New chat since last time.
        msgs, self.chat_pos = read_new_chat(self.jsonl, self.chat_pos)
        self.scorer.add_messages(signals.clean_chat(msgs, self.cfg)[0])   # no bots, !commands, sub notices

        # Loudness of every newly finished 10-s piece (decoding 10 s of audio is very cheap).
        for start, path in self.buffer.finished_segments():
            if path.name in self.done_segments:
                continue
            self.done_segments.add(path.name)
            try:
                self.scorer.add_audio(start, signals.audio_loudness(path, self.cfg["audio"]["sample_rate"]))
            except Exception as exc:
                log(self.name, f"couldn't read audio of {path.name} ({exc})")

        # Score, and turn confirmed peaks into clip jobs. The confirm time already waited until
        # the clip's end is recorded, so it's safe to cut right away.
        recorded = self.buffer.recorded_until()
        now = min(self.now(), recorded) if recorded > 0 else self.now()
        for c in self.scorer.confirmed_peaks(now):
            lag = self.now() - c.peak
            log(self.name, f"moment at {hms(c.peak)} (score {c.peak_score:.1f}, {lag:.0f} s ago)")
            if not self.save_clips:
                continue   # test mode on a channel that isn't allow-listed: never save clips
            # Cut enough around the moment for speech snapping AND the AI judge's wider window.
            a = max(0.0, c.start - before_margin(self.cfg))
            b = c.end + after_margin(self.cfg)
            try:
                source = self.dir / "sources" / f"{c.peak}.mp4"
                source_start = self.buffer.cut(a, b, source)
                self.jobs.put(ClipJob(self, c, source, source_start, b))
            except Exception as exc:
                log(self.name, f"couldn't cut the moment at {hms(c.peak)} from the recording ({exc})")

        # Keep only the last few minutes of video (and never a piece a waiting job may still need:
        # jobs get their own copy from cut(), so the buffer can go).
        self.buffer.delete_older_than(self.now() - self.cfg["live"]["buffer_minutes"] * 60)
        self.save_progress()

    def save_progress(self) -> None:
        """Chat is saved line by line already; save the loudness too, so a crash loses nothing."""
        np.save(self.dir / "audio_db.npy", self.scorer.audio)

    def close(self) -> None:
        self.chat.stop.set()
        self.buffer.stop()
        self.save_progress()
        from clipper.kick_chat import finalize
        if self.jsonl.exists():
            finalize(self.dir)   # chat.jsonl -> chat.json, the format run.py --chat reads
        log(self.name, f"stopped recording. Chat and loudness are saved in {self.dir}")


# --------------------------------------------------------------------------- clip jobs

class ClipJob:
    def __init__(self, session: Session, cand: Candidate, source: Path, source_start: float, source_end: float):
        self.session, self.cand, self.source = session, cand, source
        self.source_start, self.source_end = source_start, source_end


class ClipWorker(threading.Thread):
    """Turns confirmed moments into finished clips in the review queue, one at a time.

    One at a time on purpose: Whisper (GPU) and x264 (CPU) are the only heavy steps, and doing
    them one after another keeps the laptop usable for a game or stream running alongside.
    """

    def __init__(self, cfg: dict, jobs: queue.Queue, out_root: Path = OUT_ROOT):
        super().__init__(daemon=True)
        self.cfg, self.jobs, self.out_root = cfg, jobs, out_root
        self.busy = False
        self.transcriber = None
        self.renderer = None
        self.judge = None
        if cfg["judge"]["enabled"] and os.getenv("ANTHROPIC_API_KEY"):
            from clipper.judge import Judge
            self.judge = Judge(cfg)

    def run(self) -> None:
        while True:
            job = self.jobs.get()
            if job is None:
                return
            self.busy = True
            try:
                self.process(job)
            except Exception as exc:
                log(job.session.name, f"clip at {hms(job.cand.peak)} failed: {exc}")
            finally:
                self.busy = False

    def process(self, job: ClipJob) -> None:
        from clipper import framecheck
        from clipper.edit import ClipRenderer
        from clipper.judge import extract_frames, frame_times, judge_all
        from clipper.queue_app import connect, import_clips
        from clipper.transcribe import Transcriber, build_sentences, words_between

        s, c, cfg = job.session, job.cand, self.cfg
        t0 = time.time()

        # Transcribe the whole cut piece, snap the edges to sentences, clamp the length.
        if self.transcriber is None:
            self.transcriber = Transcriber(cfg)
            self.renderer = ClipRenderer(cfg)
        duration = job.source_end - job.source_start
        raw = self.transcriber.transcribe_window(job.source, 0.0, duration)
        for w in raw:   # file seconds -> stream seconds
            w.start += job.source_start
            w.end += job.source_start
        sentences = build_sentences(raw, cfg["whisper"]["sentence_gap"])
        c.start, c.end = snap_to_speech(c.start, c.end, c.peak, sentences, cfg["clip"])
        c.start, c.end = clamp_length(c.start, c.end, c.peak, cfg["clip"], job.source_end)
        c.start = max(c.start, job.source_start)   # we can only use what we recorded
        c.transcript = words_between(raw, c.start, c.end)

        cands_path = s.dir / "candidates.json"
        data = (json.loads(cands_path.read_text(encoding="utf-8")) if cands_path.exists() else
                {"url": s.url, "platform": s.platform, "vod_id": s.vod_id, "streamer": s.name,
                 "live": True, "signals": [], "candidates": []})
        j = cfg["judge"]
        d = c.to_dict()
        d.update({"rank": len(data["candidates"]) + 1, "peak_time": hms(c.peak),
                  "window_start": round(max(job.source_start, c.start - j["window_before"]), 2),
                  "window_end": round(min(job.source_end, c.end + j["window_after"]), 2),
                  "chat_mood": signals.mood_mix(s.scorer.messages, c.start, c.end, cfg),
                  "extra_warnings": [s.launch_warning] if s.launch_warning else []})

        # Local frame check + frames for the AI, both straight from our own recording.
        frames = []
        try:
            gray = framecheck.gray_frames(job.source, c.start - job.source_start, c.end - job.source_start)
            stats = framecheck.frame_stats(gray, cfg, None)
            d["framecheck_reasons"] = framecheck.framecheck_reasons(stats, None, cfg)
            frames = extract_frames(job.source, job.source_start,
                                    frame_times(d["window_start"], d["window_end"], c.peak, j["frames"]),
                                    j["frame_width"])
        except Exception as exc:
            log(s.name, f"couldn't look at the frames ({exc})")

        # AI judge (if ANTHROPIC_API_KEY is set): picks the cut, rates it, flags spoilers.
        judge_all([d], [raw], s.scorer.messages, s.scorer.chat_z_full(int(d["window_end"]) + 1), cfg, s.name,
                  s.title, self.judge, log=lambda text: log(s.name, text.strip()),
                  frames_per_candidate=[frames], vod_length=job.source_end)
        d["start_time"] = hms(d["start"])
        d["transcript"] = words_between(raw, d["start"], d["end"])
        d["words"] = [[round(w.start, 2), round(w.end, 2), w.text]
                      for w in raw if w.end > d["window_start"] and w.start < d["window_end"]]
        d["words_range"] = [d["window_start"], d["window_end"]]
        # Live clips are rendered one by one, so a weak moment would waste a render and your time:
        # if the AI rated it below `live.min_quality`, it's saved (for learning) but not edited.
        d["kept"] = not (d["ai"] and d["ai_quality"] < cfg["live"]["min_quality"])
        data["candidates"].append(d)
        cands_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        if not d["kept"]:
            log(s.name, f"moment at {hms(c.peak)} skipped: AI rated it {d['ai_quality']:.1f}/10 "
                        f"({d['ai'].get('reason', '')})")
            job.source.unlink(missing_ok=True)
            return

        # Stage 2 edit straight from the cut piece. Warned clips are rendered too, so you can
        # look at them, but they arrive in the queue marked with the warning (hard rule 2).
        final = self.renderer.render(s.dir, data, d, source=job.source, source_start=job.source_start)
        db = connect(self.out_root / "queue.db")
        try:
            import_clips(db, self.out_root)
        finally:
            db.close()
        job.source.unlink(missing_ok=True)
        warn = " ⚠ " + ", ".join(d["warning_reasons"]) if d["warning"] else ""
        log(s.name, f"clip ready in the review queue: {final.name} "
                    f"({time.time() - t0:.0f} s to process, {s.now() - c.peak:.0f} s after the moment){warn}")


# --------------------------------------------------------------------------- main loop

def before_margin(cfg: dict) -> float:
    """Seconds recorded before a candidate's rough start: room to snap back to a sentence
    start and for the AI judge's wider window."""
    return max(cfg["clip"]["snap_back_max"], cfg["judge"]["window_before"]) + 2


def after_margin(cfg: dict) -> float:
    """Seconds needed after a candidate's rough end before it can be cut (same reasons)."""
    return max(cfg["clip"]["snap_forward_max"], cfg["judge"]["window_after"]) + 2


def live_cfg(cfg: dict) -> dict:
    """Live mode's tweaks on top of the normal config: e.g. fewer x264 threads so a game
    running on the same laptop doesn't stutter while a clip renders."""
    cfg = json.loads(json.dumps(cfg))   # deep copy
    cfg["edit"]["threads"] = cfg["live"]["x264_threads"]
    if cfg["live"].get("whisper_model"):
        cfg["whisper"]["model"] = cfg["live"]["whisper_model"]
    return cfg


def run(cfg: dict, only: list[str] | None = None, test_channel: str | None = None) -> None:
    streamers = load_streamers()
    lv = cfg["live"]
    if test_channel:
        # Test mode (any channel): records and scores, prints the moments it would clip, and
        # saves NO clips (hard rule 3). Its recording goes to a temporary folder you can delete.
        watch = {test_channel.lower(): {"platform": "twitch"}}
        out_root = OUT_ROOT / "_live_test"
        print(f"TEST MODE on '{test_channel}': no clips are saved or queued.")
    else:
        watch = {n: s for n, s in streamers.items() if not only or n in only}
        out_root = OUT_ROOT
        if not watch:
            raise SystemExit("No streamers to watch: add them to streamers.toml (only ones who allow clipping).")
        print(f"Watching {len(watch)} allow-listed streamer(s): {', '.join(watch)}")

    checker = LiveChecker()
    jobs: queue.Queue = queue.Queue()
    worker = ClipWorker(cfg, jobs)
    worker.start()
    sessions: dict[str, Session] = {}
    next_poll = 0.0
    print("Stop with Ctrl+C. Clips go to the review queue: python -m clipper.queue_app\n")
    try:
        while True:
            if time.time() >= next_poll:
                next_poll = time.time() + lv["poll_seconds"]
                live_now: dict[str, dict] = {}
                try:
                    live_now.update(checker.twitch([n for n, s in watch.items()
                                                    if s.get("platform", "twitch") == "twitch"]))
                except SystemExit:
                    raise
                except Exception as exc:
                    print(f"[{datetime.now():%H:%M:%S}] couldn't ask Twitch who is live ({exc})")
                    live_now.update({n: {"unknown": True} for n in sessions})   # don't stop on a blip
                for n, s in watch.items():
                    if s.get("platform") == "kick":
                        try:
                            info = checker.kick(n)
                            if info:
                                live_now[n] = info
                        except Exception as exc:
                            print(f"[{datetime.now():%H:%M:%S}] couldn't ask Kick about {n} ({exc})")
                            if n in sessions:
                                live_now[n] = {"unknown": True}

                for n, info in live_now.items():
                    if info.get("unknown"):
                        continue
                    sess = sessions.get(n)
                    if sess is not None and sess.start != info["start"]:
                        sess.close()              # the stream restarted: new VOD, new folder
                        sess = None
                    if sess is None:
                        sessions[n] = Session(n, watch[n].get("platform", "twitch"), info, cfg, jobs,
                                              out_root, save_clips=not test_channel)
                for n in list(sessions):
                    if n in live_now:
                        sessions[n].offline_checks = 0
                        continue
                    # Offline twice in a row = the stream really ended (not an API hiccup).
                    sessions[n].offline_checks += 1
                    if sessions[n].offline_checks >= 2:
                        log(n, "stream ended")
                        sessions.pop(n).close()

            for sess in sessions.values():
                try:
                    sess.tick()
                except Exception as exc:
                    log(sess.name, f"problem in this round ({exc}); carrying on")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for sess in sessions.values():
            sess.close()
        waiting = jobs.qsize() + (1 if worker.busy else 0)
        if waiting:
            print(f"Finishing {waiting} clip(s) that are being made. Press Ctrl+C again to quit right away.")
            jobs.put(None)
            try:
                worker.join()
            except KeyboardInterrupt:
                print("Quit without finishing them. Their recordings are kept in each live_ folder's sources\\.")


def main() -> None:
    p = argparse.ArgumentParser(description="AutoClipper Stage 4: clip allow-listed streamers while they're live.")
    p.add_argument("--only", help="watch only these allow-listed streamers, e.g. name1,name2")
    p.add_argument("--test-channel", help="TEST: watch any Twitch channel, print moments, save no clips")
    p.add_argument("--config", type=Path, help="use a different config.toml")
    args = p.parse_args()
    load_dotenv(PROJECT_ROOT / ".env")
    cfg = live_cfg(load_config(args.config))
    only = [x.strip().lower() for x in args.only.split(",")] if args.only else None
    try:
        run(cfg, only, args.test_channel)
    except fetch.ToolMissing as exc:
        sys.exit(f"Missing tool: {exc}")


if __name__ == "__main__":
    main()
