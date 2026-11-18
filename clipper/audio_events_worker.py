"""Runs INSIDE .venv-demucs (which has PyTorch). Do not import this from the main program.

Reads a job file:   {"wavs": ["a.wav", ...], "out": "result.json"}
Writes:             {"a.wav": {"laugh": [...], "scream": [...], "music": [...], "speech": [...]}, ...}
with one probability (0..1) per second of each wav, from PANNs (a model trained on Google's
AudioSet: 527 kinds of sounds, including laughter, screaming, music and speech).

Why a separate program? PyTorch brings its own NVIDIA libraries; keeping it out of the main
environment means it can never break faster-whisper's GPU setup.
"""

import json
import sys

import numpy as np

GROUPS = {
    "laugh": ["Laughter", "Baby laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"],
    "scream": ["Screaming", "Yell", "Shout", "Battle cry", "Children shouting", "Whoop"],
    "music": ["Music"],
    "speech": ["Speech"],
}


def main() -> None:
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    import librosa
    from panns_inference import SoundEventDetection, labels
    from pathlib import Path

    ckpt = Path.home() / "panns_data" / "Cnn14_DecisionLevelMax.pth"
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sed = SoundEventDetection(checkpoint_path=str(ckpt), device=device)
    idx = {g: [labels.index(n) for n in names if n in labels] for g, names in GROUPS.items()}
    result = {}
    for wav in job["wavs"]:
        audio, _ = librosa.load(wav, sr=32000, mono=True)
        if len(audio) < 32000:
            result[wav] = {g: [] for g in GROUPS}
            continue
        frames = sed.inference(audio[None, :])[0]          # (time frames at 100 per second, 527)
        n = len(frames) // 100
        # The loudest frame in each second: a laugh or a scream is often shorter than a second.
        per_sec = frames[: n * 100].reshape(n, 100, -1).max(axis=1)
        result[wav] = {g: [round(float(v), 3) for v in per_sec[:, i].max(axis=1)] for g, i in idx.items()}
    json.dump(result, open(job["out"], "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
