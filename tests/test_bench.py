"""Benchmark logic on synthetic data: answer key, hits, tuning never peeks at the test set."""

from types import SimpleNamespace

import numpy as np

from clipper import bench
from clipper.config import load_config

CFG = load_config()


def test_human_moments_are_consensus_moments():
    clips = ([{"vod_offset": 100 + i, "duration": 30, "views": 5} for i in range(4)]       # 4 viewers, ~100 s
             + [{"vod_offset": 500, "duration": 30, "views": 9000}]                         # 1 viral clip alone
             + [{"vod_offset": 900 + i, "duration": 30, "views": 1} for i in range(2)])     # 2 viewers
    m = bench.human_moments(clips, CFG, n=5)
    assert m[0]["clippers"] == 4 and abs(m[0]["t"] - (101.5 + 18)) < 1     # most clippers first
    assert all(x["clippers"] >= 2 for x in m)                              # lone clips aren't moments
    assert len(m) == 2


def fake_vod(moments, n=3600, seed=0):
    rng = np.random.default_rng(seed)
    z = {k: rng.normal(0, 0.5, n) for k in ("chat", "unique", "hype", "audio")}
    for t in moments:
        z["hype"][t - 2:t + 3] += 6                           # hype spike at each human moment
    truth = [{"t": t, "views": 100 - i, "start": t - 20, "end": t + 10} for i, t in enumerate(moments)]
    return SimpleNamespace(id="fake", z_by_lag={6: z, 0: z}, truth=truth, all_clip_times=list(moments), n=n)


def test_evaluate_counts_hits_within_20_seconds():
    vod = fake_vod([300, 900, 1500, 2100, 2700])
    r = bench.evaluate(vod, CFG, 6)
    assert r["recall15"] == 1.0 and r["views15"] == 1.0
    assert r["n_truth"] == 5 and r["start_err"] is not None


def test_tuning_only_uses_training_vods(monkeypatch):
    seen = []
    real = bench.evaluate
    monkeypatch.setattr(bench, "evaluate", lambda v, c, lag: (seen.append(v.id), real(v, c, lag))[1])
    train = [fake_vod([300, 900], seed=1), fake_vod([1500, 2100], seed=2)]
    for i, v in enumerate(train):
        v.id = f"train{i}"
    bench.tune(train, CFG, trials=5)
    assert set(seen) == {"train0", "train1"}


def test_spearman():
    assert bench.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert bench.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
