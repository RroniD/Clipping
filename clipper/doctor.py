"""Setup checker: is everything AutoClipper needs installed and working?

Usage:  python -m clipper.doctor      (or the "Check setup" button in the review queue)

Every check prints OK or a plain-English fix. It never prints your keys.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from clipper.config import PROJECT_ROOT, load_config, load_streamers


@dataclass
class Check:
    ok: bool
    name: str
    detail: str = ""     # what we found
    fix: str = ""        # what to do if it's not OK
    important: bool = True   # False = nice to have


def _run(cmd: list[str], timeout: int = 20) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace",
                           timeout=timeout)
        return (r.stdout or "") + (r.stderr or "")
    except Exception:
        return ""


def check_tools() -> list[Check]:
    out = [Check(sys.version_info >= (3, 10), "Python", f"version {sys.version.split()[0]}",
                 "Install Python 3.11 from python.org (tick 'Add to PATH').")]
    for tool, fix in (("ffmpeg", "Open PowerShell and run: winget install Gyan.FFmpeg  (then restart the computer)"),
                      ("ffprobe", "Comes with ffmpeg: winget install Gyan.FFmpeg")):
        out.append(Check(bool(shutil.which(tool)), tool, "found" if shutil.which(tool) else "not found", fix))
    tdl = shutil.which("TwitchDownloaderCLI") or (Path("C:/Tools/TwitchDownloaderCLI.exe").exists() and "C:/Tools")
    out.append(Check(bool(tdl), "TwitchDownloaderCLI (Twitch chat)", "found" if tdl else "not found",
                     "Download the Windows CLI zip from github.com/lay295/TwitchDownloader/releases, put "
                     "TwitchDownloaderCLI.exe in C:\\Tools, and add C:\\Tools to your PATH."))
    try:
        import yt_dlp  # noqa: F401
        out.append(Check(True, "yt-dlp (downloads)", "installed"))
    except ImportError:
        out.append(Check(False, "yt-dlp (downloads)", "missing", "Run Update AutoClipper.bat (it installs it)."))
    return out


def check_gpu() -> Check:
    try:
        import ctranslate2
        n = ctranslate2.get_cuda_device_count()
    except Exception:
        n = 0
    return Check(n > 0, "Graphics card for speech-to-text", "NVIDIA GPU ready" if n else "not available",
                 "Speech-to-text will use the slower CPU. Update the NVIDIA driver (GeForce Experience) "
                 "and run Update AutoClipper.bat.", important=False)


def check_gpu_encoder() -> Check:
    """Can ffmpeg use the graphics card's video encoder (NVENC)? Makes live clips faster."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return Check(False, "Fast video encoding (graphics card)", "ffmpeg missing", "", important=False)
    out = _run([ffmpeg, "-v", "error", "-f", "lavfi", "-i", "color=black:size=256x256:duration=0.2",
                "-c:v", "h264_nvenc", "-f", "null", "-"])
    ok = "Error" not in out and "error" not in out.lower()
    return Check(ok, "Fast video encoding (graphics card)", "working" if ok else "not available",
                 "Optional: update the NVIDIA driver (GeForce Experience / nvidia.com, version 610 or newer) "
                 "and live clips are made faster. Everything works without it.", important=False)


def check_keys(online: bool = True) -> list[Check]:
    from dotenv import dotenv_values
    env = dotenv_values(PROJECT_ROOT / ".env") if (PROJECT_ROOT / ".env").exists() else {}
    out = []
    twitch = bool(env.get("TWITCH_CLIENT_ID")) and bool(env.get("TWITCH_CLIENT_SECRET"))
    detail = "filled in" if twitch else "missing"
    if twitch and online:
        try:
            from clipper.fetch import twitch_app_token
            twitch_app_token(env["TWITCH_CLIENT_ID"], env["TWITCH_CLIENT_SECRET"])
            detail = "filled in and working"
        except Exception:
            twitch, detail = False, "filled in but Twitch rejects them"
    out.append(Check(twitch, "Twitch keys (who is live, viewer clips)", detail,
                     "Make an app at dev.twitch.tv/console/apps and paste the Client ID and Secret into "
                     "the .env file in the Clipping folder."))
    ai = bool(env.get("ANTHROPIC_API_KEY"))
    detail = "filled in" if ai else "missing"
    if ai and online:
        try:
            import anthropic
            anthropic.Anthropic(api_key=env["ANTHROPIC_API_KEY"]).models.retrieve(load_config()["judge"]["model"])
            detail = "filled in and working"
        except Exception as exc:
            ai, detail = False, f"filled in but not working ({type(exc).__name__})"
    out.append(Check(ai, "AI key (judge: picks and cuts clips)", detail,
                     "Make a key at platform.claude.com > API keys and paste it after ANTHROPIC_API_KEY= in .env. "
                     "Without it clips are still made, just not judged by the AI."))
    ntfy = bool(env.get("NTFY_TOPIC"))
    out.append(Check(ntfy, "Phone notifications (ntfy)", "set up" if ntfy else "not set up",
                     "Run: python -m clipper.notify --setup  and subscribe to the topic in the ntfy app.",
                     important=False))
    return out


def check_network() -> Check:
    if sys.platform != "win32":
        return Check(True, "Wi-Fi", "not Windows: skipped", important=False)
    out = _run(["powershell", "-NoProfile", "-Command",
                "Get-NetConnectionProfile | ForEach-Object { $_.Name + '|' + $_.NetworkCategory }"])
    nets = [line.split("|") for line in out.splitlines() if "|" in line]
    public = [n for n, cat in nets if cat.strip() == "Public"]
    return Check(not public, "Wi-Fi allows your phone to connect",
                 ", ".join(f"{n} ({c.strip()})" for n, c in nets) or "unknown",
                 f"Your Wi-Fi '{public[0] if public else ''}' is set to Public, which blocks your phone. "
                 "If it's your HOME Wi-Fi: Settings > Network & internet > Wi-Fi > (your network) > "
                 "Network profile type > Private. Don't do this on school or cafe Wi-Fi.", important=False)


def check_disk(cfg: dict) -> Check:
    free = shutil.disk_usage(PROJECT_ROOT).free / 1e9
    need = cfg["live"]["min_free_gb"] * 4
    return Check(free >= need, "Free disk space", f"{free:.0f} GB free",
                 f"Less than {need:.0f} GB free. Delete old VOD folders in Clipping\\out\\ (keep what's in the queue).")


def check_setup() -> list[Check]:
    streamers = load_streamers()
    out = [Check(bool(streamers), "Streamers on your allow-list",
                 ", ".join(streamers) or "none",
                 "Add the streamers you're allowed to clip on the Settings page of the review queue.")]
    demucs = (PROJECT_ROOT / ".venv-demucs" / "Scripts" / "python.exe").exists() or \
        (PROJECT_ROOT / ".venv-demucs" / "bin" / "python").exists()
    out.append(Check(demucs, "Music removal (Demucs)", "installed" if demucs else "not installed",
                     "Optional. Run Update AutoClipper.bat to install it.", important=False))
    return out


def run_checks(online: bool = True) -> list[Check]:
    cfg = load_config()
    return [*check_tools(), check_gpu(), check_gpu_encoder(), *check_keys(online), check_network(), check_disk(cfg),
            *check_setup()]


def report(checks: list[Check]) -> str:
    lines = []
    for c in checks:
        mark = "OK " if c.ok else ("!! " if c.important else "-- ")
        lines.append(f"{mark} {c.name}: {c.detail}")
        if not c.ok and c.fix:
            lines.append(f"     -> {c.fix}")
    bad = [c for c in checks if not c.ok and c.important]
    lines.append("")
    lines.append("All important things are working." if not bad else
                 f"{len(bad)} thing(s) need fixing (marked !!). Things marked -- are optional.")
    return "\n".join(lines)


def main() -> None:
    os.chdir(PROJECT_ROOT)
    print("Checking your AutoClipper setup...\n")
    print(report(run_checks()))


if __name__ == "__main__":
    main()
