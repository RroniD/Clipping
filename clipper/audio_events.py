"""Laughter / scream / music detection for candidate moments (an extra signal next to chat,
loudness and viewer clips).

Why: "loud" isn't the same as "funny". A streamer can be loud all stream; a burst of LAUGHTER or
a real SCREAM is a much stronger sign of a clip-worthy moment, and it works even when chat is
slow (small streamers) or lagging. Music with nobody talking is also a hint of a cutscene or of
copyrighted music.

How: only the candidate windows are analysed (not the whole VOD: on this laptop's CPU that would
take ~20 minutes for a 3-hour stream). The model (PANNs, in the separate .venv-demucs) gives a
probability per second. The boost for a candidate = how much laughter/screaming there is around
its peak compared with the same windows' typical level (a z-score across all candidates of
this VOD), times the weights in config.toml.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from clipper.config import PROJECT_ROOT
from clipper.edit import demucs_python
from clipper.fetch import require_tool

WORKER = Path(__file__).with_name("audio_events_worker.py")


def available() -> bool:
    return demucs_python() is not None and (Path.home() / "panns_data" / "Cnn14_DecisionLevelMax.pth").exists()


def analyse(audio: Path, windows: list[tuple[float, float]]) -> list[dict[str, list[float]]]:
    """Per-second laugh/scream/music/speech probabilities for each (start, end) window of `audio`.
    One worker run for all windows, so the model is loaded once."""
    if not windows or not available():
        return [{} for _ in windows]
    ffmpeg = require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        wavs = []
        for i, (a, b) in enumerate(windows):
            wav = Path(tmp) / f"w{i}.wav"
            subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{max(0.0, a):.2f}", "-t", f"{b - a:.2f}",
                            "-i", str(audio), "-vn", "-ac", "1", "-ar", "32000", str(wav)], check=True)
            wavs.append(str(wav))
        job, out = Path(tmp) / "job.json", Path(tmp) / "out.json"
        job.write_text(json.dumps({"wavs": wavs, "out": str(out)}), encoding="utf-8")
        r = subprocess.run([str(demucs_python()), str(WORKER), str(job)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=str(PROJECT_ROOT))
        if r.returncode != 0 or not out.exists():
            return [{} for _ in windows]
        data = json.loads(out.read_text(encoding="utf-8"))
    return [data.get(w, {}) for w in wavs]


def event_features(probs: dict[str, list[float]], window_start: float, peak: float,
                   start: float, end: float) -> dict[str, float]:
    """Summaries for one candidate: max laugh/scream around the peak (-10 s..+8 s; reactions
    come right after the moment), and music/speech shares over the clip."""
    def span(name, a, b):
        v = probs.get(name) or []
        i, j = max(0, int(a - window_start)), max(0, int(b - window_start))
        return np.array(v[i:j]) if j > i else np.zeros(0)
    laugh, scream = span("laugh", peak - 10, peak + 8), span("scream", peak - 10, peak + 8)
    music, speech = span("music", start, end), span("speech", start, end)
    return {"laugh": round(float(laugh.max()), 3) if len(laugh) else 0.0,
            "scream": round(float(scream.max()), 3) if len(scream) else 0.0,
            "music_share": round(float((music > 0.5).mean()), 2) if len(music) else 0.0,
            "speech_share": round(float((speech > 0.5).mean()), 2) if len(speech) else 0.0}


def apply_boost(results: list[dict], cfg: dict) -> None:
    """Add laugh/scream to each candidate's peak_score as z-scores across this VOD's candidates
    (so a laugh only counts if it stands out from the other moments), and put them in the
    breakdown so the AI judge and clipper.learn can see them."""
    w = cfg["weights"]
    for name in ("laugh", "scream", "facecam_reaction"):
        vals = np.array([d.get("extra_signals", {}).get(name, 0.0) for d in results])
        if len(vals) < 3 or vals.std() == 0:
            continue
        z = (vals - np.median(vals)) / (1.4826 * np.median(np.abs(vals - np.median(vals))) + 0.05)
        for d, zi in zip(results, z):
            zc = float(np.clip(zi, 0, cfg["scoring"]["z_cap"]))
            d.setdefault("breakdown", {})[name] = round(float(zi), 2)
            d["peak_score"] = round(d["peak_score"] + float(w.get(name, 0)) * zc, 3)


def music_warning(features: dict, cfg: dict) -> str | None:
    """Music most of the clip and almost no talking: a cutscene, or copyrighted music."""
    if features.get("music_share", 0) >= cfg["audio"]["music_only_share"] and features.get("speech_share", 1) < 0.2:
        return "music with almost no talking (cutscene or copyright risk)"
    return None
