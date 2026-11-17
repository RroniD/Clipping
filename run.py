"""AutoClipper Stage 1: find the best moments in a Twitch or Kick VOD.

Usage (PowerShell, from the project folder):
    python run.py https://www.twitch.tv/videos/2245678901 --debug
    python run.py https://kick.com/somechannel/videos/<uuid> --chat out\\somechannel\\kick_live_20261117_1800\\chat.json

Output goes to out/<streamer>/<vod_id>/:
    audio.*              the VOD's audio (kept so re-runs are fast)
    chat.json            chat replay (Twitch)
    viewer_clips.json    viewer clips from the Twitch API (if keys are in .env)
    analysis/            small 360p copies of each candidate, only for looking at frames
    candidates.json      every candidate, best first; "kept" = the ones worth editing
    previews/            horizontal preview videos of the kept candidates
    debug_signals.png    with --debug: plot of every signal and the picks

How it beats watching the VOD by hand: it scores EVERY second of the stream, proposes ~40
moments, and an AI judge that sees frames, transcript, chat and viewer clips keeps the best 15.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from clipper import fetch, framecheck, signals
from clipper.config import PROJECT_ROOT, load_config, load_streamers
from clipper.judge import Judge, extract_frames, frame_times, judge_all
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


# --------------------------------------------------------------------------- chat lag per streamer

def calibrated_lag(streamer_dir: Path, vod_id: str, measured: tuple[int, float] | None, default: float) -> float:
    """Remember each VOD's measured chat lag in out/<streamer>/calibration.json and use the
    median of the trustworthy ones. (We don't edit streamers.toml: it's your file, with your
    comments in it.)"""
    path = streamer_dir / "calibration.json"
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"chat_lag": {}}
    if measured is not None:
        data["chat_lag"][vod_id] = {"lag": measured[0], "corr": round(measured[1], 3)}
        streamer_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    good = [v["lag"] for v in data["chat_lag"].values() if v["corr"] >= 0.1]
    return float(np.median(good)) if good else default


# --------------------------------------------------------------------------- main

def main() -> None:
    p = argparse.ArgumentParser(description="AutoClipper Stage 1: find clip-worthy moments in a VOD.")
    p.add_argument("url", help="Twitch or Kick VOD URL")
    p.add_argument("--chat", type=Path, help="chat file to use (needed for Kick: from clipper.kick_chat)")
    p.add_argument("--chat-offset", type=float, default=0.0,
                   help="seconds to add to every chat time if chat and VOD don't line up")
    p.add_argument("--top-n", type=int, help="override general.top_n (clips kept) from config.toml")
    p.add_argument("--config", type=Path, help="use a different config.toml")
    p.add_argument("--no-transcribe", action="store_true", help="skip Whisper (no speech snapping)")
    p.add_argument("--no-judge", action="store_true", help="skip the AI judge (Stage 1b)")
    p.add_argument("--no-previews", action="store_true", help="don't download preview clips")
    p.add_argument("--debug", action="store_true", help="save debug_signals.png")
    args = p.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    cfg = load_config(args.config)
    streamers = load_streamers()
    platform, vod_id = fetch.parse_vod_url(args.url)
    keep_n = args.top_n or cfg["general"]["top_n"]

    print("1/9 Reading VOD info")
    info = fetch.vod_info(args.url)
    streamer = check_allow_list(info, args.url, platform, streamers)
    streamer_cfg = streamers[streamer]
    out_dir = PROJECT_ROOT / "out" / streamer / vod_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"    {streamer} / {info.get('title', '')!r} -> {out_dir}")
    launch_warning = framecheck.launch_gate(framecheck.info_texts(info), framecheck.recorded_at(info), cfg)
    if launch_warning:
        print(f"    WARNING: {launch_warning}. Every clip gets the warning flag (hard rule 2).")

    print("2/9 Downloading audio only")
    audio_path = fetch.download_audio(args.url, out_dir)
    n_seconds = int(math.ceil(fetch.media_duration(audio_path)))

    print("3/9 Getting chat")
    chat_path = args.chat
    if chat_path is None and platform == "twitch":
        try:
            chat_path = fetch.download_twitch_chat(vod_id, out_dir / "chat.json")
        except Exception as exc:
            print(f"    could not get Twitch chat ({exc}); continuing with audio only")
    messages: list = []
    if chat_path:
        messages = signals.load_chat(chat_path)
        for m in messages:
            m.t += args.chat_offset
        messages, dropped = signals.clean_chat(messages, cfg)
        print(f"    {len(messages)} chat messages (ignored {dropped['bots']} from bots, "
              f"{dropped['commands']} !commands, {dropped['notices']} sub/raid notices)")
    else:
        print("    no chat for this VOD: running on audio (and viewer clips) only")

    print("4/9 Measuring audio loudness")
    loud = signals.audio_loudness(audio_path, cfg["audio"]["sample_rate"])
    audio = np.full(n_seconds, loud.min() if len(loud) else -90.0)
    audio[: min(n_seconds, len(loud))] = loud[:n_seconds]

    clips = None
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
        if clips is not None:
            print(f"    {len(clips)} viewer clips")
            # Chat lag differs per streamer (stream delay, chat speed). Viewer clips tell us
            # where moments really are, so we can measure it instead of guessing 6 s.
            measured = signals.estimate_chat_lag(messages, clips, n_seconds, cfg) if messages else None
            lag = calibrated_lag(out_dir.parent, vod_id, measured, cfg["chat"]["lag_seconds"])
            if measured:
                print(f"    chat lag measured on this VOD: {measured[0]} s (match {measured[1]:.2f}); using {lag:.0f} s")
            cfg["chat"]["lag_seconds"] = lag

    raw: dict[str, np.ndarray] = {}
    if messages:
        raw.update(signals.chat_signals(messages, n_seconds, cfg))
    raw["audio"] = audio
    if clips:
        raw["viewer_clips"] = signals.viewer_clip_signal(clips, n_seconds, cfg)

    print("5/9 Scoring and picking moments")
    z, score, candidates = find_candidates(raw, cfg, cfg["general"]["top_n_detect"])
    print(f"    {len(candidates)} candidate moments above threshold {cfg['peaks']['threshold']}")
    if clips and cfg["clip"]["viewer_clip_bounds"]:
        used = 0
        for c in candidates:
            bounds = signals.viewer_clip_bounds(clips, c.peak)
            if bounds and bounds[0] < c.peak < bounds[1] and (bounds[0] < c.start or bounds[1] > c.end):
                # Viewers' cut can only make the clip LONGER (earlier start / later end), never
                # cut away build-up the detector kept.
                c.start, c.end = min(c.start, bounds[0]), max(c.end, bounds[1])
                used += 1
        if used:
            print(f"    {used} moments extended to where viewers clipped them")

    print("6/9 Transcribing candidate windows and snapping to speech")
    jcfg = cfg["judge"]
    words_by_candidate: dict[int, list] = {}
    if not args.no_transcribe and candidates:
        transcriber = Transcriber(cfg)
        # Transcribe the whole stretch the AI may cut from (plus margin), not just the rough clip.
        pad = max(cfg["whisper"]["window_padding"], jcfg["window_before"], jcfg["window_after"]) + 2
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
    for c in candidates:
        d = c.to_dict()
        d["peak_time"] = hms(c.peak)
        d["window_start"] = round(max(0.0, c.start - jcfg["window_before"]), 2)
        d["window_end"] = round(min(float(n_seconds), c.end + jcfg["window_after"]), 2)
        if messages:
            d["chat_mood"] = signals.mood_mix(messages, c.start, c.end, cfg)
        if launch_warning:
            d["extra_warnings"] = [launch_warning]
        results.append(d)
    words_list = [words_by_candidate.get(id(c)) for c in candidates]

    print("7/9 Looking at the frames (small 360p copies of each moment)")
    frames_list = [[] for _ in results]
    if cfg["framecheck"]["enabled"] or (not args.no_judge and jcfg["enabled"]):
        hud_box = framecheck.hud_box_fractions(streamer_cfg)

        def fetch_analysis(d):
            path = out_dir / "analysis" / f"{d['peak']}_{d['window_start']:.0f}-{d['window_end']:.0f}.mp4"
            return fetch.download_section(args.url, d["window_start"], d["window_end"], path,
                                          cfg["framecheck"]["analysis_format"])

        # 4 downloads at once: each one spends most of its time waiting on Twitch, not working.
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(fetch_analysis, d) for d in results]
        stats = []
        for d, fut in zip(results, futures):
            try:
                video = fut.result()
                d["analysis"] = str(video.relative_to(out_dir))
                frames = framecheck.gray_frames(video, d["start"] - d["window_start"], d["end"] - d["window_start"])
                stats.append(framecheck.frame_stats(frames, cfg, hud_box))
                frames_list[results.index(d)] = extract_frames(
                    video, d["window_start"],
                    frame_times(d["window_start"], d["window_end"], d["peak"], jcfg["frames"]),
                    jcfg["frame_width"])
            except Exception as exc:
                print(f"    {d['peak_time']}: couldn't look at the frames ({str(exc)[-200:]})")
                stats.append(None)
        all_hud = [v for s in stats if s for v in s["hud"]]
        # The 75th percentile, not the median: if many candidates are cutscenes (HUD hidden), the
        # median would BE a hidden HUD; the upper quarter is reliably 'HUD visible'.
        hud_ref = float(np.percentile(all_hud, 75)) if all_hud else None
        for d, s in zip(results, stats):
            if s and cfg["framecheck"]["enabled"]:
                d["framecheck_reasons"] = framecheck.framecheck_reasons(s, hud_ref, cfg)
        flagged = sum(bool(d.get("framecheck_reasons")) for d in results)
        print(f"    frame check: {flagged} moment(s) look like a cutscene or loading screen")

    print("8/9 AI judge: watching, cutting and ranking the moments")
    judge = None
    if args.no_judge or not jcfg["enabled"]:
        print("    skipped (--no-judge or judge.enabled = false)")
    elif not os.getenv("ANTHROPIC_API_KEY"):
        print("    no ANTHROPIC_API_KEY in .env: skipping the AI judge")
    else:
        judge = Judge(cfg)
    judge_all(results, words_list, messages, z.get("chat"), cfg, streamer, info.get("title", ""), judge,
              frames_per_candidate=frames_list, viewer_clips=clips, vod_length=float(n_seconds))
    if judge is not None:
        cost = judge.cost()
        print(f"    AI judge used {judge.tokens_in:,} input + {judge.tokens_out:,} output tokens"
              + (f" = ${cost:.4f}" if cost is not None else ""))
    for d, words in zip(results, words_list):
        if words is not None:
            d["transcript"] = words_between(words, d["start"], d["end"])
            # Word timings for the whole window, so the editor and the queue's trim buttons
            # can make captions without running Whisper again.
            d["words"] = [[round(w.start, 2), round(w.end, 2), w.text]
                          for w in words if w.end > d["window_start"] and w.start < d["window_end"]]
            d["words_range"] = [d["window_start"], d["window_end"]]
        if platform == "twitch":
            d["vod_link"] = f"https://www.twitch.tv/videos/{vod_id}?t={hms(max(0, d['start']))}"
        d["start_time"] = hms(d["start"])

    # Best first. Keep the top `keep_n`, but give ~15% of those slots to random picks from the
    # next tier ("explore"): if those turn out as good as the top picks, the ranking needs
    # adjusting, and clipper.learn can see it. Seeded by the VOD id, so re-runs pick the same.
    order = sorted(range(len(results)), key=lambda i: results[i]["final_score"], reverse=True)
    results = [results[i] for i in order]
    candidates = [candidates[i] for i in order]
    n_explore = round(keep_n * cfg["general"]["explore_share"]) if len(results) > keep_n else 0
    kept = set(range(keep_n - n_explore))
    pool_idx = list(range(keep_n - n_explore, min(len(results), 2 * keep_n)))
    explore = set(random.Random(vod_id).sample(pool_idx, min(n_explore, len(pool_idx))))
    for i, d in enumerate(results):
        d["kept"] = i in kept or i in explore
        d["explore"] = i in explore
    final_order = [i for i in range(len(results)) if results[i]["kept"]] + \
                  [i for i in range(len(results)) if not results[i]["kept"]]
    results = [results[i] for i in final_order]
    candidates = [candidates[i] for i in final_order]
    for rank, d in enumerate(results, 1):
        d["rank"] = rank
    flagged = sum(d["warning"] for d in results if d["kept"])
    print(f"    kept {sum(d['kept'] for d in results)} of {len(results)} moments "
          f"({n_explore} explore picks); {flagged} kept clip(s) have a warning (preview names start with WARN)")

    print("9/9 Downloading previews of the kept moments")
    pad = cfg["download"]["padding"]
    for d in results:
        if args.no_previews or not d["kept"]:
            continue
        name = f"{d['rank']:02d}_{'WARN_' if d['warning'] else ''}{d['peak_time']}.mp4"
        try:
            # A preview from an earlier run may have the same name but other bounds: replace it.
            (out_dir / "previews" / name).unlink(missing_ok=True)
            path = fetch.download_section(args.url, max(0, d["start"] - pad), d["end"] + pad,
                                          out_dir / "previews" / name, cfg["download"]["preview_format"])
            d["preview"] = str(path.relative_to(out_dir))
            print(f"    {name}")
        except Exception as exc:
            print(f"    preview {name} failed: {exc}")

    candidates_path = out_dir / "candidates.json"
    candidates_path.write_text(json.dumps({"url": args.url, "platform": platform, "vod_id": vod_id,
                                           "streamer": streamer, "title": info.get("title", ""),
                                           "vod_seconds": n_seconds, "chat_lag": cfg["chat"]["lag_seconds"],
                                           "signals": sorted(raw), "candidates": results},
                                          indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {candidates_path}")

    if args.debug:
        from clipper.plot import save_debug_plot
        shown = [c for c, d in zip(candidates, results) if d["kept"]]
        png = save_debug_plot(z, score, shown, cfg["peaks"]["threshold"],
                              out_dir / "debug_signals.png", title=f"{streamer} {vod_id}")
        print(f"Saved {png}")


if __name__ == "__main__":
    try:
        main()
    except fetch.ToolMissing as exc:
        sys.exit(f"Missing tool: {exc}")
