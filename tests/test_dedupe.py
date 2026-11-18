"""Duplicate guard: one clip per moment."""

import json
from datetime import datetime, timezone

from clipper import dedupe


def clip(peak, start, end, kept=True):
    return {"peak": peak, "start": start, "end": end, "kept": kept, "peak_time": str(peak)}


def test_overlapping_cuts_keep_the_better_one():
    results = [clip(100, 80, 120), clip(130, 95, 140), clip(300, 280, 320)]   # best first
    assert dedupe.drop_overlaps(results) == 1
    assert [d["kept"] for d in results] == [True, False, True]
    assert "same moment" in results[1]["skip_reason"]


def test_small_overlap_is_fine():
    results = [clip(100, 80, 120), clip(140, 115, 160)]   # 5 s overlap of 40 s clips
    assert dedupe.drop_overlaps(results) == 0


def test_moments_already_clipped_live_are_skipped(tmp_path):
    start = datetime(2026, 11, 19, 18, 0, tzinfo=timezone.utc)
    live = tmp_path / "live_20261119_180000"
    live.mkdir()
    (live / "session.json").write_text(json.dumps({"stream_start": start.isoformat()}))
    (live / "candidates.json").write_text(json.dumps({"candidates": [{"peak": 500, "kept": True}]}))
    peaks = dedupe.live_peaks_for(tmp_path, start)
    assert peaks == [500.0]
    results = [clip(520, 500, 540), clip(900, 880, 920)]
    assert dedupe.drop_already_clipped(results, peaks) == 1
    assert results[0]["skip_reason"] == "already clipped live" and results[1]["kept"]
    # A different stream (other day) doesn't count.
    assert dedupe.live_peaks_for(tmp_path, datetime(2026, 11, 20, tzinfo=timezone.utc)) == []