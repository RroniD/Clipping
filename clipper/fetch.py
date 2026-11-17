"""Everything that downloads things: VOD info, audio, chat, viewer clips, preview sections.

Design notes
- External tools (yt-dlp, ffmpeg, TwitchDownloaderCLI) are run with `subprocess` and a
  list of arguments (never one big string). With a list, Windows paths that contain
  spaces (like "OneDrive\\Desktop") just work, and nothing gets interpreted by a shell.
- yt-dlp is started as `python -m yt_dlp` instead of `yt-dlp`. On Windows, pip often
  installs the `yt-dlp.exe` launcher into a Scripts folder that isn't on PATH. Running it
  through the same Python that runs this program always finds it.
- Every download step skips work if its output file already exists, so re-running the
  pipeline after tweaking config.toml is fast (only the scoring is redone).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

YT_DLP = [sys.executable, "-m", "yt_dlp"]
HELIX = "https://api.twitch.tv/helix"


class ToolMissing(RuntimeError):
    """Raised when an external program we need is not installed / not on PATH."""


def require_tool(name: str, install_hint: str) -> str:
    """Return the full path of an external program, or explain how to install it."""
    path = shutil.which(name)
    if not path:
        raise ToolMissing(f"'{name}' was not found on PATH. {install_hint}")
    return path


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, raising a readable error (with the tool's own output) if it fails."""
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", **kwargs)
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "").strip()[-2000:]
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(map(str, cmd))}\n{tail}")
    return result


# --------------------------------------------------------------------------- URLs

def parse_vod_url(url: str) -> tuple[str, str]:
    """Return (platform, vod_id) for a Twitch or Kick VOD URL.

    Examples
      https://www.twitch.tv/videos/2245678901          -> ("twitch", "2245678901")
      https://kick.com/somechannel/videos/<uuid>       -> ("kick", "<uuid>")
      https://kick.com/video/<uuid>                    -> ("kick", "<uuid>")
    """
    m = re.search(r"twitch\.tv/(?:[^/]+/)?videos?/(\d+)", url)
    if m:
        return "twitch", m.group(1)
    m = re.search(r"kick\.com/(?:[^/]+/)?videos?/([0-9a-fA-F-]{8,})", url)
    if m:
        return "kick", m.group(1)
    raise ValueError(f"Not a Twitch or Kick VOD URL: {url}")


# --------------------------------------------------------------------------- yt-dlp

def vod_info(url: str) -> dict:
    """Ask yt-dlp for the VOD's metadata (channel, duration, ...) without downloading it."""
    result = run(YT_DLP + ["--dump-single-json", "--skip-download", "--no-warnings", url])
    return json.loads(result.stdout)


def channel_names(info: dict) -> set[str]:
    """All the names yt-dlp might use for the channel, lowercased (used for the allow-list)."""
    keys = ("uploader_id", "uploader", "channel", "channel_id", "creator")
    return {str(info[k]).lower() for k in keys if info.get(k)}


def download_audio(url: str, out_dir: Path) -> Path:
    """Download only the audio track (a 6 h VOD is ~300 MB of audio vs ~20 GB of video)."""
    existing = sorted(out_dir.glob("audio.*"))
    existing = [p for p in existing if p.suffix not in (".part", ".ytdl")]
    if existing:
        print(f"  audio already downloaded: {existing[0].name}")
        return existing[0]
    out_dir.mkdir(parents=True, exist_ok=True)
    # "audio_only" is Twitch's audio-only format name, "bestaudio" works on most sites,
    # and "worst" is the smallest stream as a last resort (Kick often has no audio-only).
    run(YT_DLP + ["-f", "audio_only/bestaudio/worst", "--no-part", "--no-warnings",
                  "-o", str(out_dir / "audio.%(ext)s"), url])
    files = [p for p in sorted(out_dir.glob("audio.*")) if p.suffix not in (".part", ".ytdl")]
    if not files:
        raise RuntimeError("yt-dlp finished but no audio file was written")
    return files[0]


def download_section(url: str, start: float, end: float, out_path: Path, fmt: str) -> Path:
    """Download just [start, end] seconds of the VOD as a preview clip.

    --force-keyframes-at-cuts makes the cut land on the exact second instead of the
    nearest keyframe (which can be several seconds off). It costs a re-encode of this
    short piece only, which is worth it: we're judging whether clips start mid-sentence.
    """
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    run(YT_DLP + ["-f", fmt, "--download-sections", f"*{start:.2f}-{end:.2f}",
                  "--force-keyframes-at-cuts", "--merge-output-format", "mp4",
                  "--no-warnings", "-o", str(out_path), url])
    return out_path


def media_duration(path: Path) -> float:
    """Length of a media file in seconds, via ffprobe."""
    ffprobe = require_tool("ffprobe", "Install ffmpeg: winget install Gyan.FFmpeg")
    result = run([ffprobe, "-v", "error", "-show_entries", "format=duration",
                  "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(result.stdout.strip())


# --------------------------------------------------------------------------- Twitch chat

def download_twitch_chat(vod_id: str, out_path: Path) -> Path:
    """Download the full chat replay of a Twitch VOD with TwitchDownloaderCLI."""
    if out_path.exists():
        print(f"  chat already downloaded: {out_path.name}")
        return out_path
    exe = require_tool(
        "TwitchDownloaderCLI",
        "Download the Windows release from https://github.com/lay295/TwitchDownloader/releases "
        "and put TwitchDownloaderCLI.exe in a folder on PATH.")
    run([exe, "chatdownload", "--id", vod_id, "-o", str(out_path)])
    return out_path


# --------------------------------------------------------------------------- Twitch Helix (viewer clips)

def twitch_app_token(client_id: str, client_secret: str) -> str:
    """Get an app access token (client-credentials flow). Enough for reading clips."""
    r = requests.post("https://id.twitch.tv/oauth2/token", timeout=30, data={
        "client_id": client_id, "client_secret": client_secret, "grant_type": "client_credentials"})
    r.raise_for_status()
    return r.json()["access_token"]


def _helix_get(path: str, params: dict, client_id: str, token: str) -> dict:
    r = requests.get(f"{HELIX}/{path}", params=params, timeout=30,
                     headers={"Client-Id": client_id, "Authorization": f"Bearer {token}"})
    r.raise_for_status()
    return r.json()


def _parse_twitch_duration(text: str) -> int:
    """Twitch writes VOD length like '3h12m5s'. Convert to seconds."""
    parts = dict((unit, int(num)) for num, unit in re.findall(r"(\d+)([hms])", text))
    return parts.get("h", 0) * 3600 + parts.get("m", 0) * 60 + parts.get("s", 0)


def twitch_viewer_clips(vod_id: str, client_id: str, client_secret: str) -> list[dict]:
    """Return the viewer clips made during this VOD: [{vod_offset, duration, views, title}].

    Helix can't filter clips by VOD directly, so we ask for all of the channel's clips
    created between the stream's start and end (+1 h for clips made afterwards), then
    keep the ones whose video_id is this VOD.
    """
    token = twitch_app_token(client_id, client_secret)
    videos = _helix_get("videos", {"id": vod_id}, client_id, token)["data"]
    if not videos:
        raise RuntimeError(f"Twitch API doesn't know VOD {vod_id}")
    video = videos[0]
    started = datetime.fromisoformat(video["created_at"].replace("Z", "+00:00"))
    ended = started + timedelta(seconds=_parse_twitch_duration(video["duration"]) + 3600)
    ended = min(ended, datetime.now(timezone.utc))

    clips: list[dict] = []
    params = {"broadcaster_id": video["user_id"], "first": 100,
              "started_at": started.isoformat().replace("+00:00", "Z"),
              "ended_at": ended.isoformat().replace("+00:00", "Z")}
    while True:
        page = _helix_get("clips", params, client_id, token)
        for c in page["data"]:
            # vod_offset is null when Twitch couldn't place the clip in the VOD.
            if c.get("video_id") == vod_id and c.get("vod_offset") is not None:
                clips.append({"vod_offset": c["vod_offset"], "duration": c["duration"],
                              "views": c["view_count"], "title": c.get("title", "")})
        cursor = page.get("pagination", {}).get("cursor")
        if not cursor:
            break
        params["after"] = cursor
    return clips
