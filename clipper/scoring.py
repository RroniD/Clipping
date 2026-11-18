"""Combine signals into one score, pick the peaks, and work out clip start/end times.

All functions here are pure (numbers in, numbers out, no files or network), which is
why they're the most heavily unit-tested part of the project.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Candidate:
    peak: int                    # second with the highest score
    peak_score: float
    start: float                 # current clip bounds (seconds into the VOD)
    end: float
    rough_start: float = 0.0     # bounds before snapping to speech, kept for debugging
    rough_end: float = 0.0
    breakdown: dict = field(default_factory=dict)   # each signal's z-score at the peak
    transcript: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        d = asdict(self)
        d["duration"] = round(self.duration, 2)
        return d


# --------------------------------------------------------------------------- combine

def smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Moving average over a centred window (removes one-second jitter)."""
    if width <= 1 or len(x) == 0:
        return x.copy()
    return np.convolve(x, np.ones(width) / width, mode="same")


def combine(z: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    """score = sum(weight x clip(z, 0, cap)) + bonus x min(z_audio, z_chat), then smoothed.

    - Clipping at 0 ignores "quieter than usual" moments; we only care about spikes.
    - Clipping at the cap stops one crazy signal from deciding everything alone.
    - The bonus rewards moments where the streamer gets loud AND chat explodes together,
      which is the most reliable sign of a real moment.
    Signals that are missing (e.g. no chat on a Kick VOD) simply don't contribute.
    """
    weights, s = cfg["weights"], cfg["scoring"]
    cap = float(s["z_cap"])
    n = len(next(iter(z.values())))
    score = np.zeros(n)
    for name, values in z.items():
        score += float(weights.get(name, 0.0)) * np.clip(values, 0, cap)
    if "audio" in z and "chat" in z:
        both = np.minimum(np.clip(z["audio"], 0, cap), np.clip(z["chat"], 0, cap))
        score += float(s["both_bonus"]) * both
    return smooth(score, int(s["smooth_seconds"]))


def score_signals(raw: dict[str, np.ndarray], z: dict[str, np.ndarray], cfg: dict) -> np.ndarray:
    """The detector's per-second score.

    [scoring] scorer = "learned": a model trained on what viewers clipped (learned.py) scores the
    chat + audio signals; other signals (viewer clips) are added with their hand weights.
    scorer = "hand" (or no saved model, or no chat): the hand-weighted formula above.
    """
    if cfg["scoring"].get("scorer", "hand") == "learned":
        from clipper import learned
        s = learned.score(raw, cfg)
        if s is not None:
            rest = {k: v for k, v in z.items() if k not in learned.REPLACES}
            return s + combine(rest, cfg) if rest else s
    return combine(z, cfg)


# --------------------------------------------------------------------------- peaks

def pick_peaks(score: np.ndarray, threshold: float, min_gap: int, top_n: int) -> list[int]:
    """Greedy peak picking: take the highest second, block +/- min_gap around it, repeat.

    Blocking the neighbourhood stops us from returning the same moment 5 times
    (second 100, 101, 102 ... all belong to one moment).
    """
    remaining = score.astype(float).copy()
    peaks: list[int] = []
    while len(peaks) < top_n and len(remaining):
        best = int(np.argmax(remaining))
        if remaining[best] < threshold:
            break
        peaks.append(best)
        remaining[max(0, best - min_gap): best + min_gap + 1] = -np.inf
    return peaks


def rough_bounds(score: np.ndarray, peak: int, clip_cfg: dict) -> tuple[float, float]:
    """First guess at where a clip starts and ends, from the score curve alone.

    Start: `pre_roll` seconds before the peak, extended further back (up to `build_up_max`
    more) while the score is still above decay_ratio x peak, so we keep the build-up.
    End: the first second after the peak where the score falls below decay_ratio x peak
    (looking at most `tail_max` ahead), plus `post_roll` so the reaction can finish.
    """
    n = len(score)
    level = clip_cfg["decay_ratio"] * score[peak]

    start = max(0, peak - clip_cfg["pre_roll"])
    earliest = max(0, start - clip_cfg["build_up_max"])
    while start > earliest and score[start - 1] > level:
        start -= 1

    end = peak
    latest = min(n - 1, peak + clip_cfg["tail_max"])
    while end < latest and score[end + 1] >= level:
        end += 1
    end = min(n, end + clip_cfg["post_roll"])
    return float(start), float(end)


# --------------------------------------------------------------------------- speech snapping

def snap_to_speech(start: float, end: float, peak: float,
                   sentences: list[tuple[float, float]], clip_cfg: dict) -> tuple[float, float]:
    """Move the clip edges so they don't cut a sentence in half.

    `sentences` is a list of (sentence_start, sentence_end) in VOD seconds, from Whisper.
    - If a sentence is running at `start`, move start back to where it began
      (at most `snap_back_max` seconds).
    - If a sentence is running at `end`, move end forward to where it finishes
      (at most `snap_forward_max` seconds).
    - The clip may never start later than peak - peak_guard, so the moment itself
      always has a little lead-in.
    """
    new_start, new_end = start, end
    for s, e in sentences:
        if s < start < e and start - s <= clip_cfg["snap_back_max"]:
            new_start = s
        if s < end < e and e - end <= clip_cfg["snap_forward_max"]:
            new_end = e
    new_start = min(new_start, peak - clip_cfg["peak_guard"])
    return max(0.0, new_start), new_end


# --------------------------------------------------------------------------- length clamp

def clamp_length(start: float, end: float, peak: float, clip_cfg: dict,
                 vod_length: float) -> tuple[float, float]:
    """Force the clip to be between min_seconds and max_seconds long.

    Too long: keep the start (the build-up) and shorten the end, never past the payoff's
      reaction; up to max_seconds_setup (60 s) is allowed when the setup needs it.
    Too short: grow it, 60% of the missing time added before and 40% after.
    """
    lo, hi = clip_cfg["min_seconds"], clip_cfg["max_seconds"]
    length = end - start
    if length > hi:
        # Setup beats shortness: keep the start (the build-up) and shorten the END, keeping the
        # payoff and ~4 s of reaction. Only past max_seconds_setup does the start move.
        # (It used to re-centre the clip on the peak, which cut the play that led to it.)
        end = max(start + hi, min(end, peak + clip_cfg["reaction_keep"]))
        if end - start > clip_cfg["max_seconds_setup"]:
            start = end - clip_cfg["max_seconds_setup"]
    elif length < lo:
        missing = lo - length
        start -= 0.6 * missing
        end += 0.4 * missing

    # Keep inside the VOD, shifting (not shrinking) the clip if it hits an edge.
    if start < 0:
        end, start = end - start, 0.0
    if end > vod_length:
        start, end = max(0.0, start - (end - vod_length)), vod_length
    return round(start, 2), round(end, 2)


# --------------------------------------------------------------------------- whole detector

def find_candidates(raw: dict[str, np.ndarray], cfg: dict, top_n: int | None = None):
    """Raw per-second signals -> (z-scores, score, candidates with rough bounds).

    This is steps 5-9 of Stage 1 in one place, so tests can run the real detector on
    synthetic data without any downloads.
    """
    from clipper.signals import all_z_scores

    z = all_z_scores(raw, cfg)
    score = score_signals(raw, z, cfg)
    peaks = pick_peaks(score, cfg["peaks"]["threshold"], cfg["peaks"]["min_gap_seconds"],
                       top_n or cfg["general"]["top_n"])
    candidates = []
    for p in peaks:
        start, end = rough_bounds(score, p, cfg["clip"])
        candidates.append(Candidate(
            peak=p, peak_score=round(float(score[p]), 3), start=start, end=end,
            rough_start=start, rough_end=end,
            breakdown={name: round(float(values[p]), 2) for name, values in z.items()}))
    return z, score, candidates
