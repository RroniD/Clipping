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
LOOSE_SECONDS = 30      # a looser hit: tells "wrong moment" apart from "right moment, timing off"


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
                        "start": best["vod_offset"], "end": best["vod_offset"] + best["duration"],
                        # what the viewers' clips typically cover (median start / end)
                        "span": (float(np.median([c["vod_offset"] for c in g])),
                                 float(np.median([c["vod_offset"] + c["duration"] for c in g])))})
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
        # A wider answer key for TRAINING the learned scorer: every moment 2+ viewers clipped.
        self.train_moments = human_moments(clips, cfg, n=200, min_clippers=2)
        self.raw0 = None
        self.z_by_lag = {}
        for lag in lags:
            c2 = {**cfg, "chat": {**cfg["chat"], "lag_seconds": lag}}
            raw = signals.chat_signals(msgs, self.n, c2)
            raw["audio"] = loud
            if lag == 0:
                self.raw0 = raw
            self.z_by_lag[lag] = signals.all_z_scores(raw, c2)
        self._features = None

    def features(self):
        """The learned scorer's feature matrix (chat at lag 0: the model learns the lag itself)."""
        if self._features is None:
            from clipper import learned
            self._features = learned.features(self.raw0)
        return self._features


def evaluate(vod: VodData, cfg: dict, lag: int, score: np.ndarray | None = None) -> dict:
    """Detector (no viewer clips) vs the human answer key for one VOD.
    `score`: a ready per-second score (the learned scorer); default = the hand formula."""
    if score is None:
        score = combine(vod.z_by_lag[lag], cfg)
    peaks = pick_peaks(score, cfg["peaks"]["threshold"], cfg["peaks"]["min_gap_seconds"], 40)
    truth = vod.truth
    total_weight = sum(m.get("clippers", 1) for m in truth) or 1

    def hits(k, window=HIT_SECONDS):
        found = [m for m in truth if any(abs(p - m["t"]) <= window for p in peaks[:k])]
        return found

    def inside(k):
        """Our moment lies inside what the viewers clipped (+/- 5 s). The fairest test: the
        answer key's single time is only a guess (60% into the clips), the span is what they cut."""
        spans = [m.get("span", (m["start"], m["end"])) for m in truth]
        return [m for m, (a, b) in zip(truth, spans) if any(a - 5 <= p <= b + 5 for p in peaks[:k])]

    h15, h40 = hits(15), hits(40)
    loose15, loose40 = hits(15, LOOSE_SECONDS), hits(40, LOOSE_SECONDS)
    in15, in40 = inside(15), inside(40)
    errs_s, errs_e = [], []
    for m in h15:
        p = min(peaks[:15], key=lambda p: abs(p - m["t"]))
        s, e = rough_bounds(score, p, cfg["clip"])
        errs_s.append(s - m["start"])
        errs_e.append(e - m["end"])
    unclipped = [p for p in peaks[:15] if not any(abs(p - t) <= HIT_SECONDS for t in vod.all_clip_times)]
    return {"vod": vod.id, "recall15": len(h15) / len(truth) if truth else 0.0,
            "recall40": len(h40) / len(truth) if truth else 0.0,
            "loose15": len(loose15) / len(truth) if truth else 0.0,
            "loose40": len(loose40) / len(truth) if truth else 0.0,
            "inside15": len(in15) / len(truth) if truth else 0.0,
            "inside40": len(in40) / len(truth) if truth else 0.0,
            "peaks": peaks,
            # Share of all the viewers' clipping effort we caught: a moment 12 people clipped counts 12x.
            "views15": sum(m.get("clippers", 1) for m in h15) / total_weight,
            "start_err": float(np.median(errs_s)) if errs_s else None,
            "end_err": float(np.median(errs_e)) if errs_e else None,
            "unclipped15": len(unclipped), "unclipped_peaks": unclipped[:5], "n_truth": len(truth)}


def summary(rows: list[dict]) -> dict:
    keys = ("recall15", "recall40", "views15", "loose15", "loose40", "inside15", "inside40")
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


# --------------------------------------------------------------------------- learned scorer

# Fixed split by VOD, so every run compares on the same unseen VODs. The first 3 held-out VODs
# are the ones the hand weights were never tuned on; newer downloads are split alternately.
HELD_OUT = ("2913223522", "2921253522", "2928036561", "2920442556", "2916135200", "2911712860")

# Model settings tried (picked by cross-validation on the TRAINING VODs only).
GRID = [("logistic", {"C": 0.01}),
        ("boost", {"iters": 100, "leaves": 7, "min_leaf": 500}),
        ("boost", {"iters": 200, "leaves": 15, "min_leaf": 200}),
        ("boost", {"iters": 300, "leaves": 7, "min_leaf": 1000, "lr": 0.03})]


def learned_objective(rows: list[dict]) -> float:
    """Average of top-15 / top-40 hits, by both hit rules (within 20 s; inside the viewers' clip)."""
    s = summary(rows)
    return (s["recall15"] + s["recall40"] + s["inside15"] + s["inside40"]) / 4


def fit_learned(train: list[VodData], kind: str, params: dict):
    from clipper import learned
    X = np.vstack([v.features()[0] for v in train])
    ys, ws = zip(*[learned.labels(v.n, v.train_moments) for v in train])
    y, w = np.concatenate(ys), np.concatenate(ws)
    keep = w > 0
    model = learned.make_model(kind, **params)
    if kind == "logistic":
        model.fit(X[keep], y[keep], logisticregression__sample_weight=w[keep])
    else:
        model.fit(X[keep], y[keep], sample_weight=w[keep])
    return model


def learned_score(model, vod: VodData, cfg: dict) -> np.ndarray:
    from clipper import learned
    s = cfg["scoring"]
    return learned.to_score(learned.predict(model, vod.features()[0], int(s["smooth_seconds"])),
                            float(s["learned_scale"]))


def choose_learned(train: list[VodData], cfg: dict, log=print) -> tuple[str, dict, float]:
    """Leave-one-VOD-out on the training VODs: train on all but one, score that one."""
    best = None
    for kind, params in GRID:
        rows = []
        for i, v in enumerate(train):
            m = fit_learned(train[:i] + train[i + 1:], kind, params)
            rows.append(evaluate(v, cfg, 0, learned_score(m, v, cfg)))
        o = learned_objective(rows)
        log(f"  cross-validation {kind} {params}: {o:.3f}")
        if best is None or o > best[2]:
            best = (kind, params, o)
    return best


def save_model(model, kind: str, params: dict, train: list[VodData], names: list[str]) -> None:
    import joblib
    import sklearn
    from clipper import learned
    learned.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": names, "kind": kind, "params": params,
                 "trained_on": [v.id for v in train], "sklearn": sklearn.__version__}, learned.MODEL_PATH)
    learned._cache.clear()


def set_scorer(name: str) -> None:
    import re
    path = PROJECT_ROOT / "config.toml"
    text = path.read_text(encoding="utf-8")
    text = re.sub(r'(?m)^(scorer\s*=\s*)"[a-z]+"', lambda m: f'{m.group(1)}"{name}"', text, count=1)
    path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- error analysis

def _hms(t: float) -> str:
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def error_analysis(test: list[VodData], scores: dict[str, np.ndarray], cfg: dict, n: int = 10) -> list[str]:
    """The biggest misses (moments many viewers clipped that we didn't pick in the top 40) and
    the worst false picks (our top 15 where no viewer clipped anything within 20 s)."""
    misses, falses = [], []
    for v in test:
        r = evaluate(v, cfg, 0, scores[v.id])
        peaks, score, z = r["peaks"], scores[v.id], v.z_by_lag[0]
        for m in v.truth:
            a, b = m.get("span", (m["start"], m["end"]))
            if any(abs(p - m["t"]) <= HIT_SECONDS or a - 5 <= p <= b + 5 for p in peaks[:40]):
                continue
            lo, hi = max(0, int(a) - 10), min(v.n, int(b) + 10)
            best = lo + int(np.argmax(score[lo:hi]))
            rank = 1 + sum(1 for p in peaks if score[p] > score[best])
            zmax = max(("chat", "unique", "hype", "audio", "spam"), key=lambda k: z[k][lo:hi].max())
            misses.append((m["clippers"], v.id, _hms(m["t"]), rank, float(z[zmax][lo:hi].max()), zmax))
        for p in peaks[:15]:
            near = min((abs(p - t) for t in v.all_clip_times), default=9999)
            if near > HIT_SECONDS:
                falses.append((float(score[p]), v.id, _hms(p), int(near),
                               {k: round(float(z[k][p]), 1) for k in ("unique", "hype", "spam", "audio")}))
    lines = [f"Biggest misses on the held-out VODs (viewers clipped it, not in our top 40):",
             "  clippers  VOD          time      would rank   strongest signal there"]
    for c, vid, t, rank, zv, k in sorted(misses, key=lambda m: -m[0])[:n]:
        lines.append(f"  {c:>8}  {vid}  {t:>8}   #{rank:<10}  {k} z={zv:.1f}")
    lines += ["", "Worst false picks (our top 15, no viewer clip within 20 s):",
              "  score  VOD          time      nearest viewer clip   z at the pick"]
    for s, vid, t, near, zs in sorted(falses, key=lambda f: -f[0])[:n]:
        lines.append(f"  {s:5.1f}  {vid}  {t:>8}   {near:>5} s away          {zs}")
    return lines


# --------------------------------------------------------------------------- report

def run(streamer: str = "ishowspeed", apply: bool = True, log=print) -> str:
    cfg = load_config()
    folders = sorted(p for p in (OUT / streamer).iterdir()
                     if (p / "chat.json").exists() and (p / "viewer_clips.json").exists()
                     and any(p.glob("audio.*")))
    vods = []
    for f in folders:
        log(f"loading {f.name} ...")
        v = VodData(f, cfg)
        if len(v.truth) >= 10:
            vods.append(v)
        else:
            log(f"  skipped: only {len(v.truth)} moments that several viewers clipped")
    test = [v for v in vods if v.id in HELD_OUT]
    train = [v for v in vods if v.id not in HELD_OUT]
    if len(train) < 2 or not test:
        raise SystemExit("Need VODs with viewer clips: python -m clipper.bench prepare <ids>")
    lag0 = int(cfg["chat"]["lag_seconds"]) if int(cfg["chat"]["lag_seconds"]) in vods[0].z_by_lag else 6
    hand_cfg = {**cfg, "scoring": {**cfg["scoring"], "scorer": "hand"}}

    # 1. The current hand weights.
    hand_test = [evaluate(v, hand_cfg, lag0) for v in test]
    # 2. Hand weights re-tuned on the (now bigger) training set.
    log("re-tuning the hand weights on the training VODs ...")
    w, lag, _ = tune(train, hand_cfg)
    retuned_cfg = {**hand_cfg, "weights": {**hand_cfg["weights"], **w}}
    retuned_test = [evaluate(v, retuned_cfg, lag) for v in test]
    hand_better = objective(retuned_test) > objective(hand_test) + 0.01
    best_hand = retuned_test if hand_better else hand_test
    # 3. The learned scorer: settings chosen on the training VODs, then checked once on held-out.
    log("choosing the learned model (cross-validation on the training VODs) ...")
    kind, params, cv = choose_learned(train, hand_cfg, log)
    model = fit_learned(train, kind, params)
    scores = {v.id: learned_score(model, v, hand_cfg) for v in test}
    learned_test = [evaluate(v, hand_cfg, 0, scores[v.id]) for v in test]
    sh, sl = summary(best_hand), summary(learned_test)
    learned_wins = sl["recall15"] > sh["recall15"] and sl["recall40"] > sh["recall40"]

    def line(name, rows):
        s = summary(rows)
        return (f"{name:<28} {s['recall15']:>6.0%} {s['recall40']:>6.0%}   {s['loose15']:>6.0%} {s['loose40']:>6.0%}"
                f"   {s['inside15']:>6.0%} {s['inside40']:>6.0%}")

    lines = ["AUTOCLIPPER BENCHMARK", "=====================", "",
             f"{len(vods)} VODs of {streamer}. Answer key per VOD: up to 20 moments that 3+ different Twitch",
             "viewers clipped. The detector does NOT see the viewer clips (like live mode before anyone clips).",
             f"Training VODs ({len(train)}): {', '.join(v.id for v in train)}",
             f"Held-out VODs ({len(test)}, never used for tuning or training): {', '.join(v.id for v in test)}", "",
             "Share of the viewers' moments found in our top 15 / top 40, on the HELD-OUT VODs:",
             "Hit rule:                     within 20 s     within 30 s     inside the viewers' clip",
             "                              top15  top40    top15  top40    top15  top40",
             line("hand weights (current)", hand_test),
             line(f"hand weights re-tuned", retuned_test),
             line(f"learned scorer ({kind})", learned_test), ""]
    lines += [f"Learned model: {kind} {params}, picked by cross-validation on the training VODs (score {cv:.3f}).",
              ("RESULT: the learned scorer beats the best hand weights on BOTH top-15 and top-40 "
               "-> it is now the default (config.toml [scoring] scorer = \"learned\")." if learned_wins else
               "RESULT: the learned scorer did not beat the hand weights on both top-15 and top-40 -> hand weights stay."),
              (f"Re-tuned hand weights {w}, lag {lag} s: better on held-out -> written to config.toml."
               if hand_better and apply else "Re-tuned hand weights were not better on held-out -> config.toml weights unchanged."),
              ""]
    lines += ["Per held-out VOD (within 20 s, top 15 / top 40):"]
    for h, l in zip(best_hand, learned_test):
        lines.append(f"  {h['vod']}  hand {h['recall15']:>4.0%} / {h['recall40']:>4.0%}    learned "
                     f"{l['recall15']:>4.0%} / {l['recall40']:>4.0%}   ({h['n_truth']} moments)")
    lines.append("")
    final_scores = scores if learned_wins else {
        v.id: combine(v.z_by_lag[lag if hand_better else lag0], retuned_cfg if hand_better else hand_cfg) for v in test}
    lines += error_analysis(test, final_scores, hand_cfg)
    lines.append("")
    for f in folders:
        jv = judge_vs_views(f, cfg)
        if jv:
            lines.append(f"AI judge vs number of viewers who clipped the moment, on {f.name}: Spearman {jv[0]:+.2f} "
                         f"over {jv[1]} judged moments.")
    lines += ["", "How to read this: 'top-15' is the share of the most-clipped moments that are in our 15 picks.",
              "'within 30 s' vs 'within 20 s' shows how many misses are only a timing difference.",
              "'inside the viewers' clip' = our moment lies inside what they actually cut (fairest rule:",
              "the answer key's single time is a guess at 60% into their clips).",
              "Picks nobody clipped aren't necessarily wrong: viewers can't clip everything."]
    report = "\n".join(lines)
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "REPORT.txt").write_text(report + "\n", encoding="utf-8")
    strip = lambda rows: [{k: v for k, v in r.items() if k != "peaks"} for r in rows]
    (BENCH / "results.json").write_text(json.dumps({
        "hand": strip(hand_test), "retuned": strip(retuned_test), "learned": strip(learned_test),
        "weights": w, "lag": lag, "model": [kind, params], "learned_default": bool(learned_wins and apply),
        "train": [v.id for v in train], "test": [v.id for v in test]}, indent=1), encoding="utf-8")
    if apply:
        if hand_better:
            apply_settings(w, lag)
        if learned_wins:
            # The shipped model is trained on ALL VODs (held-out included): more data, same settings.
            final = fit_learned(train + test, kind, params)
            save_model(final, kind, params, train + test, vods[0].features()[1])
            set_scorer("learned")
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
