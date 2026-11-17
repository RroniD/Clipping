"""The --debug plot: every z-scored signal, the combined score, and the picked clips.

Looking at this is how you find out *why* the detector picked (or missed) a moment,
and which weight in config.toml to change.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _hms(seconds: float, _pos=None) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def save_debug_plot(z: dict[str, np.ndarray], score: np.ndarray, candidates, threshold: float,
                    out_path: Path, title: str = "") -> Path:
    import matplotlib
    matplotlib.use("Agg")  # draw to a file, no window needed
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    rows = list(z.items()) + [("SCORE", score)]
    hours = max(len(score) / 3600, 0.5)
    fig, axes = plt.subplots(len(rows), 1, sharex=True,
                             figsize=(min(60, 8 + 6 * hours), 1.6 * len(rows) + 1))
    axes = np.atleast_1d(axes)
    t = np.arange(len(score))

    for ax, (name, values) in zip(axes, rows):
        ax.plot(t, values, linewidth=0.6, color="black" if name == "SCORE" else None)
        ax.set_ylabel(name, rotation=0, ha="right", va="center")
        ax.grid(alpha=0.3)
        for rank, c in enumerate(candidates, 1):
            ax.axvspan(c.start, c.end, color="tab:orange", alpha=0.15)
            ax.axvline(c.peak, color="tab:red", linewidth=0.8)
            if name == "SCORE":
                ax.annotate(str(rank), (c.peak, values[c.peak]), textcoords="offset points",
                            xytext=(0, 4), ha="center", fontsize=8, color="tab:red")
    axes[-1].axhline(threshold, color="tab:blue", linestyle="--", linewidth=0.8)
    axes[-1].xaxis.set_major_formatter(FuncFormatter(_hms))
    axes[-1].set_xlabel("VOD time (h:mm:ss). Red line = peak, orange = clip, blue dashed = threshold")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
