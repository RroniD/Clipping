"""Transcribe ONLY the audio around each candidate, with word timestamps (faster-whisper).

Why only windows? Whisper large-v3 on a 6-hour VOD would take a long time; 15 windows
of ~1 minute take a couple of minutes. We only need speech near the clip edges anyway.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from clipper.fetch import require_tool


@dataclass
class Word:
    start: float   # seconds into the VOD
    end: float
    text: str


def _add_nvidia_dll_folders() -> None:
    """On Windows, make the CUDA DLLs from `pip install nvidia-cublas-cu12 nvidia-cudnn-cu12`
    findable. pip puts them in site-packages/nvidia/<lib>/bin, which Windows doesn't search
    by default; os.add_dll_directory adds it for this process only."""
    if sys.platform != "win32":
        return
    for base in map(Path, sys.path):
        nvidia = base / "nvidia"
        if nvidia.is_dir():
            for bin_dir in nvidia.glob("*/bin"):
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")


class Transcriber:
    """Loads the Whisper model once, and falls back to CPU if the GPU setup is broken.

    GPU problems (missing cuBLAS/cuDNN DLLs) often only show up on the first transcription,
    not when loading, so the fallback is also tried there. CLAUDE.md: don't let GPU setup
    block Stage 1.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg["whisper"]
        self.model = None
        self.on_cpu = False

    def _load(self, cpu: bool) -> None:
        from faster_whisper import WhisperModel  # heavy import, only when needed
        if cpu:
            print(f"  loading Whisper '{self.cfg['fallback_model']}' on CPU")
            self.model = WhisperModel(self.cfg["fallback_model"], device="cpu", compute_type="int8")
            self.on_cpu = True
        else:
            _add_nvidia_dll_folders()
            print(f"  loading Whisper '{self.cfg['model']}' on {self.cfg['device']}")
            self.model = WhisperModel(self.cfg["model"], device=self.cfg["device"],
                                      compute_type=self.cfg["compute_type"])

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        try:
            self._load(cpu=self.cfg["device"] == "cpu")
        except Exception as exc:
            print(f"  GPU Whisper failed to load ({exc}). Falling back to CPU.")
            self._load(cpu=True)

    def _run(self, wav: Path) -> list:
        segments, _info = self.model.transcribe(str(wav), word_timestamps=True, vad_filter=True)
        return list(segments)  # transcribe() is lazy; list() makes it actually run

    def transcribe_window(self, audio_path: Path, start: float, end: float) -> list[Word]:
        """Words spoken between start and end (VOD seconds), with VOD-based timestamps."""
        self._ensure_model()
        ffmpeg = require_tool("ffmpeg", "Install it with: winget install Gyan.FFmpeg")
        start = max(0.0, start)
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "window.wav"
            # -ss before -i seeks quickly; Whisper wants 16 kHz mono.
            subprocess.run([ffmpeg, "-v", "error", "-y", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}",
                            "-i", str(audio_path), "-ac", "1", "-ar", "16000", str(wav)], check=True)
            try:
                segments = self._run(wav)
            except Exception as exc:
                if self.on_cpu:
                    raise
                print(f"  GPU Whisper failed while running ({exc}). Falling back to CPU.")
                self._load(cpu=True)
                segments = self._run(wav)

        words = []
        for seg in segments:
            for w in seg.words or []:
                words.append(Word(start=start + w.start, end=start + w.end, text=w.word.strip()))
        return words


def build_sentences(words: list[Word], max_gap: float = 0.8) -> list[tuple[float, float]]:
    """Group words into sentences: a sentence ends at . ! ? or at a pause longer than max_gap.

    Streamers rarely speak in neat sentences, so the pause rule matters as much as the
    punctuation rule.
    """
    sentences: list[tuple[float, float]] = []
    current_start = None
    previous_end = None
    for w in words:
        if current_start is not None and previous_end is not None and w.start - previous_end > max_gap:
            sentences.append((current_start, previous_end))
            current_start = None
        if current_start is None:
            current_start = w.start
        previous_end = w.end
        if w.text.endswith((".", "!", "?")):
            sentences.append((current_start, w.end))
            current_start = None
    if current_start is not None and previous_end is not None:
        sentences.append((current_start, previous_end))
    return sentences


def words_between(words: list[Word], start: float, end: float) -> str:
    """The transcript text of the words that fall inside [start, end]."""
    return " ".join(w.text for w in words if w.end > start and w.start < end).strip()
