"""Learned scorer: features, labels, training on synthetic chat, and the hand-formula fallback."""

import numpy as np

from clipper import learned, scoring
from tests.helpers import spec_config


def fake_raw(n=1800, moments=(400, 900, 1400), seed=0):
    """Busy-ish chat with a burst of chat + hype ~5 s after each moment, and a loud streamer."""
    rng = np.random.default_rng(seed)
    raw = {k: rng.poisson(3, n).astype(float) for k in learned.CHAT_KEYS}
    raw["x_len"] = raw["chat"] * 20
    raw["audio"] = rng.normal(-30, 2, n)
    for m in moments:
        for k in ("chat", "unique", "hype", "spam"):
            raw[k][m + 3: m + 12] += 25
        raw["audio"][m - 2: m + 4] += 15
    return raw


def test_features_have_one_row_per_second_and_no_gaps():
    raw = fake_raw()
    X, names = learned.features(raw)
    assert X.shape == (1800, len(names)) and len(set(names)) == len(names)
    assert np.isfinite(X).all()


def test_features_work_without_the_extra_chat_columns():
    raw = {k: v for k, v in fake_raw().items() if not k.startswith("x_")}
    X, names = learned.features(raw)
    assert X.shape[0] == 1800 and np.isfinite(X).all()


def test_labels_cover_the_viewers_clip_and_ignore_its_edges():
    y, w = learned.labels(200, [{"t": 100, "span": (90, 120), "views": 99}])
    assert y[90:121].all() and y[:90].sum() == 0
    assert w[100] == 3.0                    # 1 + log10(100)
    assert w[80] == 0 and w[130] == 0       # fuzzy edge: no penalty either way
    assert w[10] == 1 and y[10] == 0


def test_model_trained_on_synthetic_chat_finds_new_moments():
    train = [fake_raw(seed=s, moments=(300 + 50 * s, 900, 1500 - 40 * s)) for s in range(3)]
    X = np.vstack([learned.features(r)[0] for r in train])
    ys, ws = zip(*[learned.labels(1800, [{"t": m, "span": (m - 5, m + 15)} for m in mom])
                   for mom in [(300 + 50 * s, 900, 1500 - 40 * s) for s in range(3)]])
    model = learned.make_model("boost", iters=50, leaves=7, min_leaf=20)
    model.fit(X, np.concatenate(ys), sample_weight=np.concatenate(ws))
    test = fake_raw(seed=9, moments=(250, 1000, 1600))
    score = learned.predict(model, learned.features(test)[0])
    peaks = scoring.pick_peaks(score, 0.0, 45, 3)
    assert all(any(abs(p - m) <= 25 for p in peaks) for m in (250, 1000, 1600))


def test_score_scale_is_zero_for_ordinary_seconds():
    """A model that says 0.6 almost everywhere must not look like one giant peak."""
    p = np.full(1000, 0.6)
    p[500:510] = 0.97
    s = learned.to_score(p)
    assert s[100] == 0 and s[505] > 2.0
    assert len(scoring.pick_peaks(s, 2.0, 45, 10)) == 1
    start, end = scoring.rough_bounds(s, 505, spec_config()["clip"])
    assert end - start < 40                 # the clip ends when the spike is over


def test_hand_formula_is_used_without_a_saved_model(monkeypatch):
    cfg = spec_config()
    cfg["scoring"]["scorer"] = "learned"
    monkeypatch.setattr(learned, "load", lambda *a, **k: None)
    raw = fake_raw()
    from clipper.signals import all_z_scores
    z = all_z_scores(raw, cfg)
    assert np.allclose(scoring.score_signals(raw, z, cfg), scoring.combine(z, cfg))


def test_learned_score_plus_viewer_clips(monkeypatch):
    cfg = spec_config()
    cfg["scoring"]["scorer"] = "learned"
    monkeypatch.setattr(learned, "score", lambda raw, cfg: np.ones(len(raw["chat"])))
    raw = fake_raw()
    raw["viewer_clips"] = np.zeros(1800)
    raw["viewer_clips"][500:520] = 50
    from clipper.signals import all_z_scores
    z = all_z_scores(raw, cfg)
    s = scoring.score_signals(raw, z, cfg)
    assert s[510] > s[100] >= 1.0           # model score + viewer clips on top
