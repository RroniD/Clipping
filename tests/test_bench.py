"""Benchmark logic on synthetic data: answer key, hits, tuning never peeks at the test set."""

from types import SimpleNamespace

import numpy as np

from clipper import bench
from clipper.config import load_config

CFG = load_config()


def test_human_moments_merge_and_rank_by_views():
    clips = [{"vod_offset": 100, "duration": 30, "views": 50}, {"vod_offset": 105, "duration": 30, "views": 20},
             {"vod_offset": 500, "duration": 30, "views": 60}, {"vod_offset": 900, "duration": 30, "views": 5}]
    m = bench.human_moments(clips, CFG, n=2)
    assert [x["views"] for x in m] == [70, 60]               # the two clips at ~100 s merged
    assert m[0]["start"] == 100 and m[0]["end"] == 130


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
