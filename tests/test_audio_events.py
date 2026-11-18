"""Extra signals: laughter / scream / music features, the re-ranking boost, facecam reaction."""

import numpy as np

from clipper import audio_events, layoutcheck
from clipper.config import load_config

CFG = load_config()


def test_event_features_look_around_the_peak():
    probs = {"laugh": [0.0] * 30, "scream": [0.0] * 30, "music": [0.9] * 30, "speech": [0.1] * 30}
    probs["laugh"][22] = 0.8                       # laughter 2 s after the peak (window starts at 100)
    probs["scream"][2] = 0.9                       # a scream 18 s before: too early, ignored
    f = audio_events.event_features(probs, 100.0, peak=120.0, start=105.0, end=128.0)
    assert f["laugh"] == 0.8 and f["scream"] == 0.0
    assert f["music_share"] == 1.0 and f["speech_share"] == 0.0
    assert audio_events.music_warning(f, CFG)       # music, nobody talking -> warn


def test_boost_rewards_only_moments_that_stand_out():
    results = [{"peak_score": 5.0, "extra_signals": {"laugh": v}} for v in (0.02, 0.03, 0.01, 0.9, 0.02)]
    audio_events.apply_boost(results, CFG)
    scores = [d["peak_score"] for d in results]
    assert scores[3] > 5.0 + CFG["weights"]["laugh"]          # the big laugh gets a clear boost
    assert all(s <= 5.0 + 0.5 for i, s in enumerate(scores) if i != 3)
    assert results[3]["breakdown"]["laugh"] > 3


def test_facecam_reaction_needs_a_face_and_measures_the_spike(tmp_path, monkeypatch):
    box = (0.0, 0.37, 0.24, 0.24)
    rng = np.random.default_rng(0)
    n = 40                                                     # 10 s at 4 fps, window 100-110
    frames = rng.integers(0, 255, size=(n, 180, 320, 3)).astype(np.uint8)
    frames[..., 2] = 255
    frames[..., 0] //= 3
    y0, y1, x1 = int(0.37 * 180), int(0.61 * 180), int(0.24 * 320)
    for i in range(n):
        frames[i, y0:y1, :x1] = (170, 110, 80)                # steady skin = a face
        if 20 <= i <= 23:                                      # around the peak (105 s): big movement
            frames[i, y0:y1, :x1] = (170 + 60 * (i % 2), 110, 80 + 40 * (i % 2))
    monkeypatch.setattr(layoutcheck, "rgb_frames", lambda *a, **k: frames)
    has_face, score = layoutcheck.reaction_score(tmp_path / "v.mp4", 100, 100, 110, 105, box, CFG)
    assert has_face and score > 5
    # No facecam box measured -> no signal.
    assert layoutcheck.reaction_score(tmp_path / "v.mp4", 100, 100, 110, 105, None, CFG) == (False, 0.0)
