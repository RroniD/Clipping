"""AutoClipper Stage 1: find the best moments in a Twitch or Kick VOD.

Usage (PowerShell, from the project folder):
    python run.py https://www.twitch.tv/videos/2245678901 --debug
    python run.py https://kick.com/somechannel/videos/<uuid> --chat out\\somechannel\\kick_live_20261117_1800\\chat.json

Output goes to out/<streamer>/<vod_id>/:
    audio.*              the VOD's audio (kept so re-runs are fast)
    chat.json            chat replay (Twitch)
    viewer_clips.json    viewer clips from the Twitch API (if keys are in .env)
    candidates.json      the picked moments, best first
    previews/            short horizontal preview videos, one per candidate
    debug_signals.png    with --debug: plot of every signal and the picks
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from clipper import fetch, signals
from clipper.config import PROJECT_ROOT, load_config, load_streamers
from clipper.scoring import clamp_length, find_candidates, snap_to_speech
from clipper.transcribe import Transcriber, build_sentences, words_between


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:d}h{s % 3600 // 60:02d}m{s % 60:02d}s"


def check_allow_list(info: dict, url: str, platform: str, streamers: dict) -> str:
    """Hard rule 3: refuse to clip anyone who isn't in streamers.toml. Returns the name."""
    names = fetch.channel_names(info)
    if platform == "kick":  # kick.com/<channel>/videos/<id>: the channel is in the URL too
        parts = url.split("kick.com/")[-1].split("/")
        if len(parts) > 1 and parts[0] not in ("video", "videos"):
            names.add(parts[0].lower())
    for name in names:
        if name in streamers:
            return name
    raise SystemExit(
        f"Refusing to clip: channel {sorted(names) or '(unknown)'} is not in streamers.toml.\n"
        "Only add streamers who allow or pay for clipping.")


def main() -> None:
    p = argparse.ArgumentParser(description="AutoClipper Stage 1: find clip-worthy moments in a VOD.")
    p.add_argument("url", help="Twitch or Kick VOD URL")
    p.add_argument("--chat", type=Path, help="chat file to use (needed for Kick: from clipper.kick_chat)")
    p.add_argument("--chat-offset", type=float, default=0.0,
                   help="seconds to add to every chat time if chat and VOD don't line up")
    p.add_argument("--top-n", type=int, help="override general.top_n from config.toml")
    p.add_argument("--config", type=Path, help="use a different config.toml")
    p.add_argument("--no-transcribe", action="store_true", help="skip Whisper (no speech snapping)")
    p.add_argument("--no-previews", action="store_true", help="don't download preview clips")
    p.add_argument("--debug", action="store_true", help="save debug_signals.png")
    args = p.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_config(args.config)
    streamers = load_streamers()
    platform, vod_id = fetch.parse_vod_url(args.url)

    print("1/7 Reading VOD info")
    info = fetch.vod_info(args.url)
    streamer = check_allow_list(info, args.url, platform, streamers)
    out_dir = PROJECT_ROOT / "out" / streamer / vod_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"    {streamer} / {info.get('title', '')!r} -> {out_dir}")

    print("2/7 Downloading audio only")
    audio_path = fetch.download_audio(args.url, out_dir)
    n_seconds = int(math.ceil(fetch.media_duration(audio_path)))

    print("3/7 Getting chat")
    chat_path = args.chat
    if chat_path is None and platform == "twitch":
        try:
            chat_path = fetch.download_twitch_chat(vod_id, out_dir / "chat.json")
        except Exception as exc:
            print(f"    could not get Twitch chat ({exc}); continuing with audio only")
    raw: dict[str, np.ndarray] = {}
    if chat_path:
        messages = signals.load_chat(chat_path)
        for m in messages:
            m.t += args.chat_offset
        print(f"    {len(messages)} chat messages")
        raw.update(signals.chat_signals(messages, n_seconds, cfg))
    else:
        print("    no chat for this VOD: running on audio (and viewer clips) only")

    print("4/7 Measuring audio loudness")
    loud = signals.audio_loudness(audio_path, cfg["audio"]["sample_rate"])
    audio = np.full(n_seconds, loud.min() if len(loud) else -90.0)
    audio[: min(n_seconds, len(loud))] = loud[:n_seconds]
    raw["audio"] = audio

    if platform == "twitch":
        cid, secret = os.getenv("TWITCH_CLIENT_ID"), os.getenv("TWITCH_CLIENT_SECRET")
        clips_path = out_dir / "viewer_clips.json"
        if clips_path.exists():
            clips = json.loads(clips_path.read_text(encoding="utf-8"))
        elif cid and secret:
            try:
                clips = fetch.twitch_viewer_clips(vod_id, cid, secret)
                clips_path.write_text(json.dumps(clips, indent=1), encoding="utf-8")
            except Exception as exc:
                print(f"    could not get viewer clips ({exc})")
                clips = []
        else:
            print("    no TWITCH_CLIENT_ID / TWITCH_CLIENT_SECRET in .env: skipping viewer clips")
            clips = None
        if clips is not None:
            print(f"    {len(clips)} viewer clips")
            raw["viewer_clips"] = signals.viewer_clip_signal(clips, n_seconds, cfg)

    print("5/7 Scoring and picking peaks")
    z, score, candidates = find_candidates(raw, cfg, args.top_n)
    print(f"    {len(candidates)} candidates above threshold {cfg['peaks']['threshold']}")

    print("6/7 Transcribing candidate windows and snapping to speech")
    words_by_candidate = {}
    if not args.no_transcribe and candidates:
        transcriber = Transcriber(cfg)
        pad = cfg["whisper"]["window_padding"]
        for i, c in enumerate(candidates, 1):
            print(f"    [{i}/{len(candidates)}] {hms(c.peak)}")
            words = transcriber.transcribe_window(audio_path, c.start - pad, c.end + pad)
            sentences = build_sentences(words, cfg["whisper"]["sentence_gap"])
            c.start, c.end = snap_to_speech(c.start, c.end, c.peak, sentences, cfg["clip"])
            words_by_candidate[id(c)] = words
    for c in candidates:
        c.start, c.end = clamp_length(c.start, c.end, c.peak, cfg["clip"], n_seconds)
        if id(c) in words_by_candidate:
            c.transcript = words_between(words_by_candidate[id(c)], c.start, c.end)

    results = []
    for rank, c in enumerate(candidates, 1):
        d = c.to_dict()
        d["rank"] = rank
        d["peak_time"] = hms(c.peak)
        d["start_time"] = hms(c.start)
        if platform == "twitch":
            d["vod_link"] = f"https://www.twitch.tv/videos/{vod_id}?t={hms(max(0, c.start))}"
        results.append(d)

    print("7/7 Downloading preview sections")
    pad = cfg["download"]["padding"]
    for d in results:
        if args.no_previews:
            break
        name = f"{d['rank']:02d}_{d['peak_time']}.mp4"
        try:
            path = fetch.download_section(args.url, max(0, d["start"] - pad), d["end"] + pad,
                                          out_dir / "previews" / name, cfg["download"]["preview_format"])
            d["preview"] = str(path.relative_to(out_dir))
            print(f"    {name}")
        except Exception as exc:
            print(f"    preview {name} failed: {exc}")

    candidates_path = out_dir / "candidates.json"
    candidates_path.write_text(json.dumps({"url": args.url, "platform": platform, "vod_id": vod_id,
                                           "streamer": streamer, "vod_seconds": n_seconds,
                                           "signals": sorted(raw), "candidates": results},
                                          indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {candidates_path}")

    if args.debug:
        from clipper.plot import save_debug_plot
        png = save_debug_plot(z, score, candidates, cfg["peaks"]["threshold"],
                              out_dir / "debug_signals.png", title=f"{streamer} {vod_id}")
        print(f"Saved {png}")


if __name__ == "__main__":
    try:
        main()
    except fetch.ToolMissing as exc:
        sys.exit(f"Missing tool: {exc}")
