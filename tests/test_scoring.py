"""Unit tests for the pure functions: z-scores, peaks, bounds, snapping, clamping."""

import numpy as np

from clipper.config import load_config
from clipper.scoring import clamp_length, combine, pick_peaks, rough_bounds, snap_to_speech
from clipper.signals import is_hype, robust_z, viewer_clip_signal
from clipper.transcribe import Word, build_sentences

CFG = load_config(None)
CLIP = CFG["clip"]


def test_robust_z_flat_signal_uses_floor():
    x = np.zeros(1000)
    x[500] = 3
    z = robust_z(x, 300, floor=3.0)
    assert z[500] == 1.0          # MAD is 0, so we divide by the floor only
    assert np.all(z[:400] == 0)


def test_robust_z_ignores_slow_growth():
    """A stream getting steadily busier shouldn't look like one big spike."""
    x = np.linspace(10, 100, 3000)
    z = robust_z(x, 300, floor=1.0)
    assert np.abs(z[200:-200]).max() < 0.5


def test_pick_peaks_blocks_neighbours_and_respects_threshold():
    score = np.zeros(1000)
    score[100], score[110], score[500], score[900] = 10, 9, 5, 2
    peaks = pick_peaks(score, threshold=2.5, min_gap=45, top_n=10)
    assert peaks == [100, 500]    # 110 is blocked by 100, 900 is below threshold


def test_pick_peaks_top_n():
    score = np.zeros(1000)
    for i, t in enumerate(range(50, 1000, 100)):
        score[t] = 10 + i
    assert len(pick_peaks(score, 2.5, 45, top_n=3)) == 3


def test_rough_bounds_follow_decay():
    score = np.zeros(300)
    score[90:131] = 5          # build-up and reaction around the peak
    score[120] = 10
    start, end = rough_bounds(score, 120, CLIP)
    assert start == 90         # extended back past pre_roll while score > 35% of peak
    assert end == 130 + CLIP["post_roll"]


def test_rough_bounds_caps():
    score = np.full(500, 9.0)
    score[250] = 10
    start, end = rough_bounds(score, 250, CLIP)
    assert start == 250 - CLIP["pre_roll"] - CLIP["build_up_max"]
    assert end == 250 + CLIP["tail_max"] + CLIP["post_roll"]


def test_snap_to_speech_moves_to_sentence_edges():
    sentences = [(95.0, 103.0), (118.0, 126.0)]
    start, end = snap_to_speech(100, 121, peak=110, sentences=sentences, clip_cfg=CLIP)
    assert (start, end) == (95.0, 126.0)


def test_snap_to_speech_limits():
    sentences = [(80.0, 103.0), (118.0, 140.0)]   # too far back / too far forward
    start, end = snap_to_speech(100, 121, peak=110, sentences=sentences, clip_cfg=CLIP)
    assert (start, end) == (100, 121)


def test_snap_never_starts_after_peak_guard():
    start, _ = snap_to_speech(109, 130, peak=110, sentences=[], clip_cfg=CLIP)
    assert start == 110 - CLIP["peak_guard"]


def test_clamp_too_long_keeps_the_build_up():
    """Shortening a long clip cuts the END (after the reaction), never the setup at the start."""
    start, end = clamp_length(0, 100, peak=20, clip_cfg=CLIP, vod_length=1000)
    assert start == 0 and end == CLIP["max_seconds"]
    # Payoff late in the clip: it may run up to max_seconds_setup to keep both setup and payoff.
    start, end = clamp_length(0, 100, peak=50, clip_cfg=CLIP, vod_length=1000)
    assert start == 0 and end == 50 + CLIP["reaction_keep"]
    # Only beyond max_seconds_setup does the start move.
    start, end = clamp_length(0, 100, peak=70, clip_cfg=CLIP, vod_length=1000)
    assert end == 70 + CLIP["reaction_keep"] and end - start == CLIP["max_seconds_setup"]


def test_clamp_too_short_grows_60_40():
    start, end = clamp_length(100, 110, peak=105, clip_cfg=CLIP, vod_length=1000)
    missing = CLIP["min_seconds"] - 10
    assert (start, end) == (100 - 0.6 * missing, 110 + 0.4 * missing)


def test_clamp_stays_inside_vod():
    start, end = clamp_length(0, 5, peak=2, clip_cfg=CLIP, vod_length=1000)
    assert start == 0 and end == CLIP["min_seconds"]


def test_combine_bonus_when_audio_and_chat_spike_together():
    z = {"chat": np.array([0, 4, 0, 0, 0.0]), "audio": np.array([0, 4, 0, 0, 0.0])}
    cfg = load_config(None)
    cfg["scoring"]["smooth_seconds"] = 1
    score = combine(z, cfg)
    assert score[1] == 1.0 * 4 + 0.7 * 4 + 0.5 * 4


def test_viewer_clip_bump_position_and_weight():
    sig = viewer_clip_signal([{"vod_offset": 100, "duration": 30, "views": 99}], 300, CFG)
    assert int(np.argmax(sig)) == 118                  # 100 + 0.6 x 30
    assert abs(sig.max() - 3.0) < 1e-9                 # 1 + log10(100)


def test_hype_detection():
    words = {w.lower() for w in CFG["chat"]["hype_words"]}
    phrases = CFG["chat"]["hype_phrases"]
    for text in ["KEKW", "KEKWWW", "OMEGALUL OMEGALUL", "?????", "WWWW", "💀", "CLIP IT", "lul that was crazy"]:
        assert is_hype(text, words, phrases, 0.5) or text == "lul that was crazy", text
    for text in ["hello everyone", "what mission is this", ""]:
        assert not is_hype(text, words, phrases, 0.5), text


def test_build_sentences_by_punctuation_and_pause():
    words = [Word(0, 0.5, "okay"), Word(0.6, 1.0, "go."), Word(1.2, 1.5, "wait"),
             Word(3.0, 3.4, "what"), Word(3.5, 3.9, "happened")]
    assert build_sentences(words, max_gap=0.8) == [(0, 1.0), (1.2, 1.5), (3.0, 3.9)]
