"""Learned scorer: a small model that turns the per-second chat/audio signals into one score.

Why: the hand formula (score = sum of weight x z-score) can only say "more chat = better".
A model trained on what Twitch viewers actually clipped can learn patterns the formula can't,
for example "chat goes QUIET, then explodes 10 s later", "lots of ??? = confusion moment",
"many different people, short messages", or "the streamer gets loud BEFORE chat reacts".

How it's trained (python -m clipper.bench learn): on past VODs, every second near a moment
that several viewers clipped is a "yes" (weighted more for moments with more views), every
other second a "no". The model sees ~150 numbers per second (below) and learns which mix
predicts "yes". It's checked on VODs it never saw before it may replace the hand formula.

How it's used: scoring.score_signals() calls score() when config [scoring] scorer = "learned"
and models/scorer.joblib exists. Peak picking, clip bounds, the AI judge... stay the same.
Viewer clips (when there are any) are still added on top with their hand weight, because
the model never sees them (it's trained to find moments WITHOUT them, like in live mode).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from clipper.config import PROJECT_ROOT
from clipper.signals import robust_z

MODEL_PATH = PROJECT_ROOT / "models" / "scorer.joblib"

# Chat signals the model reads (each a 5-second window sum, see signals.chat_signals).
CHAT_KEYS = ("chat", "unique", "hype", "spam", "newcomers", "caps", "x_question",
             "x_funny", "x_hype", "x_tense", "x_confusion", "x_fail")
# Hand-formula signals the model replaces (anything else, e.g. viewer_clips, is still added).
REPLACES = set(CHAT_KEYS) | {"audio", "x_len"}
FLOOR = 1.0
BASELINE = 300
LOOKAHEAD = 30          # the model looks this many seconds past a second (chat reacts late)


def _shift(x: np.ndarray, d: int) -> np.ndarray:
    """Value d seconds later (d > 0) or earlier (d < 0), repeating the edge value."""
    if d == 0:
        return x
    if d > 0:
        return np.concatenate([x[d:], np.full(d, x[-1])])
    return np.concatenate([np.full(-d, x[0]), x[:d]])


def _mean(x: np.ndarray, width: int) -> np.ndarray:
    if width <= 1:
        return x
    return np.convolve(x, np.ones(width) / width, mode="same")


def _max(x: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    padded = np.pad(x, (half, half), mode="edge")
    return np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1).max(axis=1)


def features(raw: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """One row of numbers per second of the stream. Returns (matrix, column names)."""
    n = len(raw["audio"]) if "audio" in raw else len(raw["chat"])
    cols: list[np.ndarray] = []
    names: list[str] = []

    def add(name, arr):
        names.append(name)
        cols.append(np.asarray(arr, dtype=np.float32))

    for k in CHAT_KEYS:
        x = raw.get(k, np.zeros(n))
        s15, s30 = _mean(x, 15), _mean(x, 30)
        z5, z15, z30 = robust_z(x, BASELINE, FLOOR), robust_z(s15, BASELINE, FLOOR), robust_z(s30, BASELINE, FLOOR)
        for d in (0, 5, 10, 20):                     # chat AFTER this second: the reaction
            add(f"{k}_z5_+{d}", _shift(z5, d))
            add(f"{k}_z15_+{d}", _shift(z15, d))
        add(f"{k}_z30_+10", _shift(z30, 10))
        add(f"{k}_slope", _shift(z15, 10) - _shift(z15, -10))   # how fast it's rising
        add(f"{k}_level", np.log1p(_shift(s15, 5)))               # how busy chat is at all
    chat15 = _shift(_mean(raw.get("chat", np.zeros(n)), 15), 5) + 1.0
    for k in ("unique", "hype", "spam", "caps", "x_question", "x_funny", "x_hype", "x_tense",
              "x_confusion", "x_fail"):
        add(f"{k}_share", _shift(_mean(raw.get(k, np.zeros(n)), 15), 5) / chat15)
    msg_len = _shift(_mean(raw.get("x_len", np.zeros(n)), 15), 5) / chat15
    add("msg_len", msg_len)
    add("msg_len_z", robust_z(msg_len, BASELINE, 2.0))

    a = raw.get("audio", np.full(n, -60.0))
    az = robust_z(_mean(a, 3), BASELINE, 2.0)
    for d in (-10, -5, 0, 5, 10):
        add(f"audio_z_{d:+d}", _shift(az, d))
    add("audio_max10_z", robust_z(_max(a, 11), BASELINE, 2.0))
    add("audio_slope", _shift(az, 5) - _shift(az, -5))
    add("audio_spread", _mean(np.abs(a - _mean(a, 10)), 10))
    add("audio_level", a - np.median(a))
    return np.column_stack(cols), names


def labels(n: int, moments: list[dict], margin: float = 15) -> tuple[np.ndarray, np.ndarray]:
    """(y, weight) per second. y = 1 for seconds inside what viewers clipped (the clips' median
    span), weighted by 1 + log10(1 + views). Seconds just outside (within `margin`) get weight 0:
    clip edges are fuzzy, so the model shouldn't be punished for scoring them high or low."""
    y = np.zeros(n, dtype=np.int8)
    w = np.ones(n, dtype=np.float32)
    t = np.arange(n)
    for m in moments:
        a, b = m.get("span", (m["t"] - 7, m["t"] + 7))
        w[(t >= a - margin) & (t <= b + margin)] = 0.0
    for m in moments:
        a, b = m.get("span", (m["t"] - 7, m["t"] + 7))
        inside = (t >= a) & (t <= b)
        y[inside] = 1
        w[inside] = 1.0 + math.log10(1 + m.get("views", 0))
    return y, w


def make_model(kind: str, **params):
    """'logistic' (a weighted sum, like the hand formula but learned) or 'boost' (gradient-boosted
    decision trees: can learn 'if this AND that' rules)."""
    if kind == "logistic":
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        return make_pipeline(StandardScaler(), LogisticRegression(C=params.get("C", 0.1), max_iter=2000,
                                                                  class_weight="balanced"))
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=params.get("iters", 200), learning_rate=params.get("lr", 0.05),
                                          max_leaf_nodes=params.get("leaves", 15),
                                          min_samples_leaf=params.get("min_leaf", 200),
                                          l2_regularization=1.0, class_weight="balanced", random_state=0)


def predict(model, X: np.ndarray, smooth_seconds: int = 3) -> np.ndarray:
    p = model.predict_proba(X)[:, 1]
    return _mean(p, smooth_seconds)


# --------------------------------------------------------------------------- use in the pipeline

_cache: dict = {}


def load(path: Path = MODEL_PATH):
    """The saved model (None if there is none or it can't be read)."""
    if path in _cache:
        return _cache[path]
    model = None
    if path.exists():
        try:
            import joblib
            model = joblib.load(path)
        except Exception as e:                     # e.g. scikit-learn not installed
            print(f"    (learned scorer unavailable: {e}; using the hand-tuned weights)")
    _cache[path] = model
    return model


def score(raw: dict[str, np.ndarray], cfg: dict) -> np.ndarray | None:
    """Per-second score from the saved model, on the same scale as the hand formula (0-10).
    None = no model or no chat (e.g. a Kick VOD without recorded chat): use the hand formula."""
    saved = load()
    if saved is None or "chat" not in raw or "audio" not in raw:
        return None
    X, names = features(raw)
    if names != saved["features"]:
        print("    (learned scorer was trained on other features: retrain with python -m clipper.bench learn)")
        return None
    s = cfg["scoring"]
    return float(s.get("learned_scale", 10.0)) * predict(saved["model"], X, int(s["smooth_seconds"]))
