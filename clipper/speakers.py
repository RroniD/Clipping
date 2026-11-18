"""Who is talking? Different caption colours for the streamer and for other people.

In co-streams and IRL clips several people talk. One caption colour for everyone makes a
back-and-forth hard to follow on mute, so the main voice (whoever talks most in the clip:
almost always the streamer) keeps white captions and other voices get a different colour.

The voice grouping runs in the separate .venv-demucs (SpeechBrain), like the other PyTorch
models, so it can't break the main setup. If it isn't installed or fails, captions are simply
all one colour, as before.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from clipper.config import PROJECT_ROOT
from clipper.edit import demucs_python
from clipper.fetch import require_tool

WORKER = Path(__file__).with_name("speakers_worker.py")


def available() -> bool:
    py = demucs_python()
    if py is None:
        return False
    r = subprocess.run([str(py), "-c", "import speechbrain"], capture_output=True)
    return r.returncode == 0


def sentence_spans(words: list[dict], gap: float = 0.8) -> list[tuple[int, int]]:
    """Group word indexes into sentences (a pause or . ! ? ends one): [(first, last)]."""
    spans, start = [], 0
    for i, w in enumerate(words):
        last = i == len(words) - 1
        ends = w["text"].rstrip().endswith((".", "!", "?")) or (not last and words[i + 1]["start"] - w["end"] > gap)
        if ends or last:
            spans.append((start, i))
            start = i + 1
    return spans


def main_speaker_labels(spans: list[tuple[int, int]], labels: list[int], words: list[dict]) -> list[int]:
    """Per word: 0 = the main voice (most talk time), 1 = someone else. Unknown -> 0."""
    talk: dict[int, float] = {}
    for (a, b), lab in zip(spans, labels):
        if lab >= 0:
            talk[lab] = talk.get(lab, 0.0) + words[b]["end"] - words[a]["start"]
    main = max(talk, key=talk.get) if talk else 0
    out = [0] * len(words)
    for (a, b), lab in zip(spans, labels):
        if lab >= 0 and lab != main:
            for i in range(a, b + 1):
                out[i] = 1
    return out


def label_words(source: Path, offset: float, duration: float, words: list[dict], threshold: float = 0.55) -> list[int]:
    """Speaker (0 = main, 1 = other) for each word of the clip. All 0 if it can't tell."""
    spans = sentence_spans(words)
    if len(spans) < 2 or not available():
        return [0] * len(words)
    ffmpeg = require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "clip.wav"
        subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{offset:.3f}", "-t", f"{duration:.3f}", "-i", str(source),
                        "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
        job, out = Path(tmp) / "job.json", Path(tmp) / "out.json"
        segs = [[max(0.0, words[a]["start"]), words[b]["end"]] for a, b in spans]
        job.write_text(json.dumps({"wav": str(wav), "segments": segs, "threshold": threshold, "out": str(out)}),
                       encoding="utf-8")
        r = subprocess.run([str(demucs_python()), str(WORKER), str(job)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=str(PROJECT_ROOT))
        if r.returncode != 0 or not out.exists():
            return [0] * len(words)
        labels = json.loads(out.read_text(encoding="utf-8"))["labels"]
    return main_speaker_labels(spans, labels, words)
