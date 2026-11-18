"""Shared test helpers."""

from clipper.config import load_config


def spec_config() -> dict:
    """The config with the ORIGINAL design values (CLAUDE.md): 6 s chat lag and the starting
    weights. The synthetic detector tests plant moments the way the design describes them, so they
    must not depend on numbers that `python -m clipper.bench` tuned for one streamer."""
    cfg = load_config()
    cfg["chat"]["lag_seconds"] = 6
    cfg["weights"].update({"chat": 1.0, "unique": 0.8, "hype": 1.2, "audio": 0.7, "viewer_clips": 1.5,
                           "spam": 0.6, "newcomers": 0.0, "caps": 0.0})
    # The hand formula: the learned scorer was trained on real Speed chat timing, not on these
    # synthetic moments (it has its own tests in test_learned.py).
    cfg["scoring"]["scorer"] = "hand"
    return cfg