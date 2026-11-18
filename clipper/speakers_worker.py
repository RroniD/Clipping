"""Runs INSIDE .venv-demucs (PyTorch + SpeechBrain). Do not import this from the main program.

Job file: {"wav": "clip.wav", "segments": [[start, end], ...], "threshold": 0.55, "out": "labels.json"}
Writes:   {"labels": [0, 0, 1, 0, ...]}  one speaker number per segment (-1 = too short to tell)

How: a "voice fingerprint" (ECAPA speaker embedding, a 192-number summary of how a voice sounds)
for every sentence, then sentences whose fingerprints are similar (cosine similarity above the
threshold) are grouped as the same speaker.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    job = json.load(open(sys.argv[1], encoding="utf-8"))
    import numpy as np
    import torch
    import torchaudio
    from speechbrain.inference.speaker import EncoderClassifier

    from speechbrain.utils.fetching import LocalStrategy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # LocalStrategy.COPY: Windows doesn't allow normal users to create symlinks, which
    # SpeechBrain uses by default for its downloaded model files.
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                           savedir=str(Path.home() / "panns_data" / "ecapa"),
                                           run_opts={"device": device}, local_strategy=LocalStrategy.COPY)
    wav, sr = torchaudio.load(job["wav"])
    wav = wav.mean(dim=0)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    embs, idx = [], []
    for i, (a, b) in enumerate(job["segments"]):
        if b - a < 0.8:
            continue                       # too short to recognise a voice reliably
        piece = wav[int(a * sr): int(b * sr)]
        with torch.no_grad():
            e = model.encode_batch(piece.unsqueeze(0).to(device)).squeeze().cpu().numpy()
        embs.append(e / (np.linalg.norm(e) + 1e-9))
        idx.append(i)
    labels = [-1] * len(job["segments"])
    centroids: list = []
    for i, e in zip(idx, embs):
        sims = [float(np.dot(e, c / (np.linalg.norm(c) + 1e-9))) for c in centroids]
        if sims and max(sims) >= job.get("threshold", 0.55):
            k = int(np.argmax(sims))
            centroids[k] = centroids[k] + e
        else:
            k = len(centroids)
            centroids.append(e.copy())
        labels[i] = k
    json.dump({"labels": labels}, open(job["out"], "w", encoding="utf-8"))


if __name__ == "__main__":
    main()
