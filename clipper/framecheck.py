"""Local spoiler / cutscene checks that don't need the AI (CLAUDE.md hard rule 2: no spoilers).

Two checks:
1. Launch gate: GTA 6 footage recorded BEFORE the official launch is pre-release / leaked
   footage, which Rockstar takes down. Every clip from such a VOD gets the warning flag.
2. Frame check: looks at 1 frame per second of a candidate (small grayscale frames, so it's
   fast) for things that mean "this is a cutscene or not gameplay":
   - letterbox: black bars at the top AND bottom (GTA cutscenes are cinematic widescreen)
   - HUD missing: the minimap corner loses its usual detail (cutscenes hide the HUD).
     Only when a streamer has `hud_box` in streamers.toml, because every stream layout puts
     the game (and its minimap) somewhere else.
   - blank: an (almost) single-colour frame for a few seconds = loading or black screen

Any hit -> warning flag with the reason, so the clip is never posted without you looking.
These are heuristics: they can be wrong both ways, which is why the AI also checks frames.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from clipper.fetch import require_tool

FRAME_W, FRAME_H = 320, 180


# --------------------------------------------------------------------------- launch gate

def mentions_gta6(texts: list[str], cfg: dict) -> bool:
    patterns = [re.compile(p, re.I) for p in cfg["launch"]["title_patterns"]]
    return any(p.search(t or "") for t in texts for p in patterns)


def recorded_at(info: dict) -> datetime | None:
    """When the VOD was recorded, from yt-dlp's info (timestamp, else upload_date)."""
    if info.get("timestamp"):
        return datetime.fromtimestamp(float(info["timestamp"]), tz=timezone.utc)
    if info.get("upload_date"):
        return datetime.strptime(str(info["upload_date"]), "%Y%m%d").replace(tzinfo=timezone.utc)
    return None


def launch_gate(texts: list[str], when: datetime | None, cfg: dict) -> str | None:
    """A warning reason if this is GTA 6 footage from before the launch, else None.

    texts: stream title, game/category names, chapter names. If we can't tell WHEN it was
    recorded but it is GTA 6, we warn too: better a false alarm than a takedown.
    """
    if not mentions_gta6(texts, cfg):
        return None
    launch = datetime.fromisoformat(cfg["launch"]["gta6_us_launch_utc"])
    if when is None:
        return "GTA 6 footage with unknown recording date (could be pre-release)"
    if when < launch:
        return f"pre-release GTA 6 footage (recorded {when:%Y-%m-%d}, before the {launch:%Y-%m-%d} launch)"
    return None


def info_texts(info: dict) -> list[str]:
    """Everything in yt-dlp's VOD info that could name the game."""
    texts = [info.get("title", ""), info.get("description", "") or ""]
    texts += [c.get("title", "") for c in info.get("chapters") or []]
    texts += list(info.get("categories") or []) + list(info.get("tags") or [])
    return texts


# --------------------------------------------------------------------------- frame check

def gray_frames(video: Path, start: float, end: float) -> np.ndarray:
    """1 frame per second of [start, end] (seconds into the file) as a (n, 180, 320) uint8 array."""
    ffmpeg = require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    result = subprocess.run([ffmpeg, "-v", "error", "-ss", f"{max(0.0, start):.2f}", "-t", f"{end - start:.2f}",
                             "-i", str(video), "-vf", f"fps=1,scale={FRAME_W}:{FRAME_H},format=gray",
                             "-f", "rawvideo", "-"], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg couldn't read frames from {video.name}")
    n = len(result.stdout) // (FRAME_W * FRAME_H)
    return np.frombuffer(result.stdout[: n * FRAME_W * FRAME_H], dtype=np.uint8).reshape(n, FRAME_H, FRAME_W)


def _longest_run(flags) -> int:
    longest = run = 0
    for f in flags:
        run = run + 1 if f else 0
        longest = max(longest, run)
    return longest


def detail(region: np.ndarray) -> float:
    """How much fine detail a region has (mean absolute difference between neighbouring pixels).
    A minimap is full of lines and icons; a cutscene shot of a sky or a face is much smoother."""
    r = region.astype(np.int16)
    return float(np.abs(np.diff(r, axis=0)).mean() + np.abs(np.diff(r, axis=1)).mean())


def frame_stats(frames: np.ndarray, cfg: dict, hud_box: tuple[float, float, float, float] | None) -> dict:
    """Per-clip numbers: longest letterbox run, longest blank run, and HUD detail per frame."""
    fc = cfg["framecheck"]
    band = max(1, int(FRAME_H * fc["letterbox_share"]))
    letterbox, blank, hud = [], [], []
    for f in frames:
        top, bottom, middle = f[:band].mean(), f[-band:].mean(), f[band:-band].mean()
        # Both bars dark, but a picture in between (otherwise it's just a dark frame).
        letterbox.append(top < fc["letterbox_luma"] and bottom < fc["letterbox_luma"]
                         and middle > fc["letterbox_luma"] + 15)
        blank.append(float(f.std()) < fc["blank_std"])
        if hud_box is not None:
            x, y, w, h = hud_box
            region = f[int(y * FRAME_H): int((y + h) * FRAME_H), int(x * FRAME_W): int((x + w) * FRAME_W)]
            hud.append(detail(region) if region.size else 0.0)
    return {"letterbox_seconds": _longest_run(letterbox), "blank_seconds": _longest_run(blank), "hud": hud}


def framecheck_reasons(stats: dict, hud_reference: float | None, cfg: dict) -> list[str]:
    """Turn frame_stats into warning reasons. hud_reference = the typical minimap detail in this
    VOD (median over all candidates), so 'missing' is judged against this streamer's own HUD."""
    fc = cfg["framecheck"]
    reasons = []
    if stats["letterbox_seconds"] >= fc["letterbox_seconds"]:
        reasons.append(f"possible cutscene: cinematic black bars for {stats['letterbox_seconds']} s")
    if stats["blank_seconds"] >= fc["blank_seconds"]:
        reasons.append(f"loading/black screen for {stats['blank_seconds']} s")
    if hud_reference and stats["hud"]:
        missing = [d < fc["hud_missing_ratio"] * hud_reference for d in stats["hud"]]
        run = _longest_run(missing)
        if run >= fc["hud_missing_seconds"]:
            reasons.append(f"possible cutscene: game HUD/minimap hidden for {run} s")
    return reasons


def hud_box_fractions(streamer_cfg: dict) -> tuple[float, float, float, float] | None:
    """`hud_box = [x, y, w, h]` in streamers.toml (pixels of the measured frame) -> fractions."""
    box = streamer_cfg.get("hud_box")
    if not box:
        return None
    fw, fh = (float(v) for v in streamer_cfg.get("frame", [1920, 1080]))
    x, y, w, h = (float(v) for v in box)
    return (x / fw, y / fh, w / fw, h / fh)
