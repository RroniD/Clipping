"""Benchmark: does AutoClipper find the moments that human clippers find?

Usage (PowerShell, from the project folder):
    python -m clipper.bench prepare 2928036561 2921253522 ...   # download audio, chat, viewer clips
    python -m clipper.bench run                                  # measure + tune, writes out\\bench\\REPORT.txt

The idea: viewers already told us which moments were clip-worthy - they clipped them on Twitch.
The 20 most-viewed viewer clips of a VOD are our "answer key". We run the detector WITHOUT the
viewer-clip signal (otherwise it would be copying the answers) and measure:
  - recall@15 / recall@40: how many of those 20 moments are in our top 15 / top 40 (hit = our
    peak within +/-20 s of the moment)
  - views-weighted hit rate: the same, but a moment clipped by viewers 50,000 times counts more
  - start/end error: how far our rough cut is from where the viewers cut (median seconds)
  - "our picks nobody clipped": hidden gems or false alarms
Then the signal weights and chat lag are tuned on half of the VODs and checked on the other
half; config.toml only changes if the held-out VODs improve (no fooling ourselves).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from clipper import fetch, signals
from clipper.config import PROJECT_ROOT, load_config
from clipper.scoring import combine, pick_peaks, rough_bounds

OUT = PROJECT_ROOT / "out"
BENCH = OUT / "bench"
HIT_SECONDS = 20


# --------------------------------------------------------------------------- data

def prepare(vod_ids: list[str], streamer: str = "ishowspeed") -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    cid, secret = os.getenv("TWITCH_CLIENT_ID"), os.getenv("TWITCH_CLIENT_SECRET")
    for vod in vod_ids:
        url = f"https://www.twitch.tv/videos/{vod}"
        d = OUT / streamer / vod
        d.mkdir(parents=True, exist_ok=True)
        print(f"{vod}: audio", flush=True)
        fetch.download_audio(url, d)
        print(f"{vod}: chat", flush=True)
        fetch.download_twitch_chat(vod, d / "chat.json")
        if not (d / "viewer_clips.json").exists():
            print(f"{vod}: viewer clips", flush=True)
            clips = fetch.twitch_viewer_clips(vod, cid, secret)
            (d / "viewer_clips.json").write_text(json.dumps(clips, indent=1), encoding="utf-8")
        print(f"{vod}: ready", flush=True)


def human_moments(clips: list[dict], cfg: dict, n: int = 20, merge: float = 30, min_clippers: int = 3) -> list[dict]:
    """The answer key: moments that SEVERAL viewers clipped (consensus), most-clipped first.

    Why not simply "the most-viewed clips"? On a big channel most viewer clips are personal
    (a few dozen views, titles like "ii"), and views depend on who shared a clip. Many different
    people clipping the same 30 seconds is the clearest sign a moment was clip-worthy.
    Clips within `merge` s of each other are one moment; it needs `min_clippers` clips (lowered
    to 2 if that leaves fewer than 10 moments). Its time is the median of the clips' moments.
    """
    centre = cfg["signals"]["viewer_clip_center"]
    groups: list[list[dict]] = []
    for c in sorted(clips, key=lambda c: c["vod_offset"] + centre * c["duration"]):
        t = c["vod_offset"] + centre * c["duration"]
        if groups and t - (groups[-1][-1]["vod_offset"] + centre * groups[-1][-1]["duration"]) <= merge:
            groups[-1].append(c)
        else:
            groups.append([c])
    moments = []
    for g in groups:
        ts = [c["vod_offset"] + centre * c["duration"] for c in g]
        best = max(g, key=lambda c: c.get("views", 0))
        moments.append({"t": float(np.median(ts)), "clippers": len(g), "views": sum(c.get("views", 0) for c in g),
                        "start": best["vod_offset"], "end": best["vod_offset"] + best["duration"]})
    need = min_clippers if sum(m["clippers"] >= min_clippers for m in moments) >= 10 else 2
    moments = [m for m in moments if m["clippers"] >= need]
    return sorted(moments, key=lambda m: (-m["clippers"], -m["views"]))[:n]


class VodData:
    """Everything the detector needs for one VOD, computed once (chat signals per chat lag)."""

    def __init__(self, folder: Path, cfg: dict, lags=(0, 3, 6, 9, 12, 15)):
        self.folder = folder
        self.id = folder.name
        audio = next(p for p in folder.glob("audio.*") if p.suffix not in (".part", ".ytdl"))
        cache = folder / "bench_loudness.npy"
        if cache.exists():
            loud = np.load(cache)
        else:
            loud = signals.audio_loudness(audio, cfg["audio"]["sample_rate"])
            np.save(cache, loud)
        self.n = len(loud)
        msgs, _ = signals.clean_chat(signals.load_chat(folder / "chat.json"), cfg)
        clips = json.loads((folder / "viewer_clips.json").read_text(encoding="utf-8"))
        self.truth = human_moments(clips, cfg)
        self.all_clip_times = [c["vod_offset"] + cfg["signals"]["viewer_clip_center"] * c["duration"] for c in clips]
        self.z_by_lag = {}
        for lag in lags:
            c2 = {**cfg, "chat": {**cfg["chat"], "lag_seconds": lag}}
            raw = signals.chat_signals(msgs, self.n, c2)
            raw["audio"] = loud
            self.z_by_lag[lag] = signals.all_z_scores(raw, c2)


def evaluate(vod: VodData, cfg: dict, lag: int) -> dict:
    """Detector (no viewer clips) vs the human answer key for one VOD."""
    z = vod.z_by_lag[lag]
    score = combine(z, cfg)
    peaks = pick_peaks(score, cfg["peaks"]["threshold"], cfg["peaks"]["min_gap_seconds"], 40)
    truth = vod.truth
    total_weight = sum(m.get("clippers", 1) for m in truth) or 1

    def hits(k):
        found = [m for m in truth if any(abs(p - m["t"]) <= HIT_SECONDS for p in peaks[:k])]
        return found

    h15, h40 = hits(15), hits(40)
    errs_s, errs_e = [], []
    for m in h15:
        p = min(peaks[:15], key=lambda p: abs(p - m["t"]))
        s, e = rough_bounds(score, p, cfg["clip"])
        errs_s.append(s - m["start"])
        errs_e.append(e - m["end"])
    unclipped = [p for p in peaks[:15] if not any(abs(p - t) <= HIT_SECONDS for t in vod.all_clip_times)]
    return {"vod": vod.id, "recall15": len(h15) / len(truth) if truth else 0.0,
            "recall40": len(h40) / len(truth) if truth else 0.0,
            # Share of all the viewers' clipping effort we caught: a moment 12 people clipped counts 12x.
            "views15": sum(m.get("clippers", 1) for m in h15) / total_weight,
            "start_err": float(np.median(errs_s)) if errs_s else None,
            "end_err": float(np.median(errs_e)) if errs_e else None,
            "unclipped15": len(unclipped), "unclipped_peaks": unclipped[:5], "n_truth": len(truth)}


def summary(rows: list[dict]) -> dict:
    keys = ("recall15", "recall40", "views15")
    return {k: round(float(np.mean([r[k] for r in rows])), 3) for k in keys}


def objective(rows: list[dict]) -> float:
    """What tuning maximises: mostly the views-weighted top-15 hits, plus top-40 recall."""
    s = summary(rows)
    return 0.6 * s["views15"] + 0.4 * s["recall40"]


# --------------------------------------------------------------------------- tuning

SIGNALS = ("chat", "unique", "hype", "audio", "spam", "newcomers", "caps")


def tune(train: list[VodData], cfg: dict, trials: int = 300, seed: int = 1) -> tuple[dict, int, float]:
    """Random search over the chat-based signal weights and the chat lag, on the training VODs."""
    rng = random.Random(seed)
    base_lag = int(cfg["chat"]["lag_seconds"])
    best_w = {k: cfg["weights"][k] for k in SIGNALS}
    best_lag = base_lag if base_lag in train[0].z_by_lag else 6
    best = objective([evaluate(v, cfg, best_lag) for v in train])
    for _ in range(trials):
        w = {k: round(rng.uniform(0.0, 2.0), 2) for k in SIGNALS}
        lag = rng.choice(list(train[0].z_by_lag))
        c2 = {**cfg, "weights": {**cfg["weights"], **w}}
        score = objective([evaluate(v, c2, lag) for v in train])
        if score > best:
            best, best_w, best_lag = score, w, lag
    return best_w, best_lag, best


def spearman(a, b) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def judge_vs_views(folder: Path, cfg: dict) -> tuple[float, int] | None:
    """Spearman correlation between the AI's quality rating of each judged candidate and how many
    views viewer clips of that moment got (0 if nobody clipped it). Uses an existing run.py
    result, so it costs nothing."""
    cands = folder / "candidates.json"
    clips_path = folder / "viewer_clips.json"
    if not cands.exists() or not clips_path.exists():
        return None
    data = json.loads(cands.read_text(encoding="utf-8"))["candidates"]
    clips = json.loads(clips_path.read_text(encoding="utf-8"))
    judged = [d for d in data if d.get("ai")]
    if len(judged) < 5:
        return None
    centre = cfg["signals"]["viewer_clip_center"]
    # How many viewers clipped this moment (the same "consensus" idea as the answer key).
    clippers = [sum(1 for c in clips if abs(c["vod_offset"] + centre * c["duration"] - d["peak"]) <= HIT_SECONDS)
                for d in judged]
    quality = [d.get("ai_quality", d["ai"]["score"]) for d in judged]
    return spearman(quality, clippers), len(judged)


# --------------------------------------------------------------------------- report

def run(streamer: str = "ishowspeed", apply: bool = True) -> str:
    cfg = load_config()
    folders = sorted(p for p in (OUT / streamer).iterdir()
                     if (p / "chat.json").exists() and (p / "viewer_clips.json").exists()
                     and any(p.glob("audio.*")))
    vods = []
    for f in folders:
        print(f"loading {f.name} ...", flush=True)
        v = VodData(f, cfg)
        if len(v.truth) >= 10:
            vods.append(v)
    if len(vods) < 2:
        raise SystemExit("Need at least 2 VODs with viewer clips: python -m clipper.bench prepare <ids>")
    lag0 = int(cfg["chat"]["lag_seconds"]) if int(cfg["chat"]["lag_seconds"]) in vods[0].z_by_lag else 6
    before = [evaluate(v, cfg, lag0) for v in vods]
    half = len(vods) // 2
    train, test = vods[:half], vods[half:]
    w, lag, train_score = tune(train, cfg)
    tuned_cfg = {**cfg, "weights": {**cfg["weights"], **w}}
    test_before = objective([evaluate(v, cfg, lag0) for v in test])
    test_after = objective([evaluate(v, tuned_cfg, lag) for v in test])
    after = [evaluate(v, tuned_cfg, lag) for v in vods]
    improved = test_after > test_before + 0.01

    lines = ["AUTOCLIPPER BENCHMARK", "=====================", "",
             f"Answer key: up to 20 moments per VOD that 3+ different Twitch viewers clipped (most-clipped first) ({len(vods)} VODs of {streamer}).",
             "The detector does NOT see the viewer clips. A 'hit' = our moment within 20 s of theirs.", "",
             "VOD          top-15 hits  top-40 hits  clippers covered by top 15   start/end error   our top-15 nobody clipped"]

    def row(r):
        se = "-" if r["start_err"] is None else f"{r['start_err']:+.0f}s/{r['end_err']:+.0f}s"
        return (f"{r['vod']:<12} {r['recall15']:>10.0%} {r['recall40']:>12.0%} {r['views15']:>20.0%}   "
                f"{se:>16}   {r['unclipped15']:>6}")
    lines += ["BEFORE tuning (current settings):"] + [row(r) for r in before]
    s0 = summary(before)
    lines += [f"AVERAGE      {s0['recall15']:>10.0%} {s0['recall40']:>12.0%} {s0['views15']:>20.0%}", ""]
    lines += [f"Tuning on {', '.join(v.id for v in train)}; checking on {', '.join(v.id for v in test)} "
              "(VODs the tuning never saw)."]
    lines += [f"Best settings found: weights {w}, chat lag {lag} s",
              f"Held-out score: before {test_before:.3f} -> after {test_after:.3f} "
              f"({'better: settings applied to config.toml' if improved and apply else 'not better enough: config.toml unchanged'})", ""]
    lines += ["AFTER tuning:"] + [row(r) for r in after]
    s1 = summary(after)
    lines += [f"AVERAGE      {s1['recall15']:>10.0%} {s1['recall40']:>12.0%} {s1['views15']:>20.0%}", ""]
    for f in folders:
        jv = judge_vs_views(f, cfg)
        if jv:
            lines.append(f"AI judge vs number of viewers who clipped the moment, on {f.name}: Spearman {jv[0]:+.2f} over {jv[1]} judged moments "
                         "(+1 = the AI ranks moments exactly like viewers' views; 0 = unrelated).")
    lines += ["", "How to read this: 'top-15 hits' is the share of the 20 most-clipped moments that are in our 15 picks.",
              "Human clippers usually post 3-15 clips per stream, so top-15 is the number that matters most.",
              "'Nobody clipped' picks aren't necessarily wrong: viewers can't clip everything."]
    report = "\n".join(lines)
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "REPORT.txt").write_text(report + "\n", encoding="utf-8")
    (BENCH / "results.json").write_text(json.dumps({"before": before, "after": after, "weights": w, "lag": lag,
                                                    "test_before": test_before, "test_after": test_after,
                                                    "applied": bool(improved and apply)}, indent=1), encoding="utf-8")
    if improved and apply:
        apply_settings(w, lag)
    return report


def apply_settings(weights: dict, lag: int) -> None:
    """Write the tuned weights and chat lag into config.toml (only the numbers, comments kept)."""
    import re
    path = PROJECT_ROOT / "config.toml"
    text = path.read_text(encoding="utf-8")
    for k, v in weights.items():
        text = re.sub(rf"(?m)^({k}\s*=\s*)[0-9.]+", lambda m: f"{m.group(1)}{v}", text, count=1)
    text = re.sub(r"(?m)^(lag_seconds\s*=\s*)[0-9.]+", lambda m: f"{m.group(1)}{lag}", text, count=1)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark the detector against what Twitch viewers clipped.")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("vods", nargs="+")
    rp = sub.add_parser("run")
    rp.add_argument("--no-apply", action="store_true", help="measure and tune, but don't change config.toml")
    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args.vods)
    else:
        print(run(apply=not args.no_apply))


if __name__ == "__main__":
    main()
