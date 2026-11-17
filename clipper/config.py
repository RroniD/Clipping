"""Load config.toml and streamers.toml, filling in defaults for anything missing.

Why defaults in code AND a config file?
  config.toml is where you tune things. The defaults below mean an old or partial
  config.toml still works, and the tests can run without any config file at all.
"""

from __future__ import annotations

import copy
try:
    import tomllib  # built into Python 3.11+
except ModuleNotFoundError:  # Python 3.10: same reader, installed from pip as "tomli"
    import tomli as tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULTS: dict = {
    "general": {"top_n": 15},
    "chat": {
        "lag_seconds": 6,
        "smooth_seconds": 5,
        "baseline_seconds": 300,
        "hype_ratio": 0.5,
        "hype_words": [
            "kekw", "omegalul", "lul", "lmao", "lmfao", "pog", "pogchamp", "poggers", "pogu",
            "w", "ww", "www", "monkas", "pepelaugh", "icant", "sadge", "holy", "wtf", "bro",
            "💀", "😂", "🤣", "😭", "🔥",
        ],
        "hype_phrases": ["clip it", "clip that", "no way", "oh my god", "omg"],
    },
    "audio": {"sample_rate": 8000, "baseline_seconds": 300},
    "signals": {
        "floor": {"chat": 3.0, "unique": 2.0, "hype": 1.0, "audio": 2.0, "viewer_clips": 0.5},
        "viewer_clip_sigma": 8,
        "viewer_clip_center": 0.6,
    },
    "weights": {"chat": 1.0, "unique": 0.8, "hype": 1.2, "audio": 0.7, "viewer_clips": 1.5},
    "scoring": {"z_cap": 8, "both_bonus": 0.5, "smooth_seconds": 3},
    "peaks": {"threshold": 2.5, "min_gap_seconds": 45},
    "clip": {
        "min_seconds": 15,
        "max_seconds": 60,
        "target_seconds": 30,
        "pre_roll": 12,
        "build_up_max": 20,
        "post_roll": 3,
        "tail_max": 45,
        "decay_ratio": 0.35,
        "keep_before_ratio": 0.45,
        "snap_back_max": 8,
        "snap_forward_max": 6,
        "peak_guard": 3,
    },
    "whisper": {
        "model": "large-v3",
        "device": "cuda",
        "compute_type": "int8_float16",
        "window_padding": 10,
        "fallback_model": "small",
        "sentence_gap": 0.8,
    },
    "judge": {
        "enabled": True,
        "model": "claude-opus-5-5",
        "effort": "low",
        "max_candidates": 20,
        "max_tokens": 4000,
        "chat_sample": 25,
        "fallbacks": True,
        "warn_on": ["low", "high"],
        "cutscene_quiet_seconds": 12,
        "chat_quiet_z": 0.0,
        "neutral_score": 5,
    },
    "download": {
        "padding": 2,
        "preview_format": "bv*[height<=720]+ba/b[height<=720]/b",
    },
    "edit": {
        "top_n": 10,
        "source_format": "bv*[height<=1080]+ba/b[height<=1080]/b",
        "fps": 30,
        "preset": "veryfast",
        "crf": 20,
        "facecam_share": 0.35,
        "blur_strength": 10,
        "font": "Arial Black",
        "caption_size": 84,
        "caption_max_words": 3,
        "caption_max_chars": 16,
        "caption_gap": 0.6,
        "caption_uppercase": True,
        "caption_y_no_facecam": 0.70,
        "caption_y_crop": 0.62,
        "hook": True,
        "hook_size": 64,
        "hook_seconds": 2.2,
        "hook_y": 0.17,
        "credit": True,
        "credit_size": 34,
        "ui_bottom_share": 0.20,
        "loudness_lufs": -14,
        "duck_music_db": 0,
    },
    "live": {
        "poll_seconds": 60,
        "check_seconds": 5,
        "window_seconds": 900,
        "warmup_seconds": 240,
        "edge_seconds": 60,
        "confirm_seconds": 45,
        "max_peaks_per_check": 10,
        "buffer_minutes": 6,
        "segment_seconds": 10,
        "buffer_format": "best[height<=1080]/best",
        "x264_threads": 4,
        "whisper_model": "",
    },
    "queue": {
        "host": "0.0.0.0",
        "port": 8000,
        "trim_step": 2,
        "min_seconds": 5,
        "max_seconds": 90,
    },
}


def _merge(base: dict, override: dict) -> dict:
    """Recursively copy `override` on top of `base` (nested tables are merged, not replaced)."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path | None = None) -> dict:
    """Return the full config as a nested dict, e.g. cfg["peaks"]["threshold"]."""
    path = Path(path) if path else PROJECT_ROOT / "config.toml"
    if not path.exists():
        return copy.deepcopy(DEFAULTS)
    with path.open("rb") as f:  # tomllib wants binary mode
        user_cfg = tomllib.load(f)
    return _merge(DEFAULTS, user_cfg)


def load_streamers(path: Path | None = None) -> dict[str, dict]:
    """Return the allow-list as {login_name_lowercase: settings}."""
    path = Path(path) if path else PROJECT_ROOT / "streamers.toml"
    if not path.exists():
        return {}
    with path.open("rb") as f:
        data = tomllib.load(f)
    return {name.lower(): (info or {}) for name, info in data.get("streamers", {}).items()}
