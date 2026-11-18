"""Never queue two clips of the same moment.

Two ways duplicates happen:
1. Inside one VOD: the AI may lengthen two nearby moments until their cuts overlap.
   -> if two kept clips overlap by more than half of the shorter one, keep the better-ranked.
2. Live mode clipped a moment, and later you run the VOD of the SAME stream through run.py.
   Live clips are timed in seconds since the stream started, the same as the VOD, so
   a VOD moment within +/- 45 s of a live clip's peak is the same moment -> skip it.
Posting the same moment twice splits its views and campaigns may reject the second one.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path


def overlap_share(a: dict, b: dict) -> float:
    """How much two cuts overlap, as a share of the shorter one (0..1)."""
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    shorter = min(a["end"] - a["start"], b["end"] - b["start"])
    return max(0.0, inter) / shorter if shorter > 0 else 0.0


def drop_overlaps(results: list[dict], max_share: float = 0.5) -> int:
    """results must be best first. Marks later overlapping kept clips as not kept. Returns how many."""
    kept: list[dict] = []
    dropped = 0
    for d in results:
        if not d.get("kept"):
            continue
        twin = next((k for k in kept if overlap_share(d, k) > max_share), None)
        if twin:
            d["kept"] = False
            d["skip_reason"] = f"same moment as the better clip at {twin.get('peak_time', '')}"
            dropped += 1
        else:
            kept.append(d)
    return dropped


def live_peaks_for(streamer_dir: Path, vod_start: datetime | None, tolerance_min: float = 10) -> list[float]:
    """Peaks (stream seconds) that live mode already clipped from the stream this VOD belongs to."""
    if vod_start is None:
        return []
    peaks = []
    for session in streamer_dir.glob("live_*/session.json"):
        info = json.loads(session.read_text(encoding="utf-8"))
        start = datetime.fromisoformat(info["stream_start"])
        if abs(start - vod_start) > timedelta(minutes=tolerance_min):
            continue
        cands = session.parent / "candidates.json"
        if cands.exists():
            peaks += [float(d["peak"]) for d in json.loads(cands.read_text(encoding="utf-8"))["candidates"]
                      if d.get("kept", True)]
    return peaks


def drop_already_clipped(results: list[dict], live_peaks: list[float], gap: float = 45) -> int:
    """Marks kept VOD moments that live mode already clipped as not kept. Returns how many."""
    dropped = 0
    for d in results:
        if d.get("kept") and any(abs(d["peak"] - p) <= gap for p in live_peaks):
            d["kept"] = False
            d["skip_reason"] = "already clipped live"
            dropped += 1
    return dropped
